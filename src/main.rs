//! parakeet-server — OpenAI-compatible ASR server backed by parakeet-rs.
//!
//! Exposes:
//!   POST /v1/audio/transcriptions  (OpenAI-compatible)
//!   GET  /v1/models
//!   GET  /health

mod audio;
mod error;

use std::sync::{Arc, Mutex};

use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::{HeaderMap, Method},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::{Parser, ValueEnum};
use error::AppError;
use parakeet_rs::{ExecutionConfig, ExecutionProvider, Nemotron, Parakeet, ParakeetEOU, ParakeetTDT, TimestampMode, Transcriber};
use serde::Serialize;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ValueEnum)]
enum ModelType {
    Ctc,
    Tdt,
    Eou,
    Nemotron,
}

#[derive(Debug, Clone, ValueEnum)]
enum Provider {
    Cpu,
    Webgpu,
}

#[derive(Parser, Debug)]
#[command(
name = "parakeet-server",
about = "OpenAI-compatible ASR server powered by NVIDIA Parakeet via parakeet-rs"
)]
struct Args {
    #[arg(long, value_enum, default_value = "ctc", env = "PARAKEET_MODEL_TYPE")]
    model_type: ModelType,

    #[arg(long, default_value = "./models", env = "PARAKEET_MODEL_DIR")]
    model_dir: String,

    #[arg(long, default_value = "0.0.0.0", env = "PARAKEET_HOST")]
    host: String,

    #[arg(long, default_value = "8000", env = "PARAKEET_PORT")]
    port: u16,

    #[arg(long, default_value = "0", env = "PARAKEET_MAX_DURATION")]
    max_duration: u64,

    #[arg(long, env = "PARAKEET_API_KEY")]
    api_key: Option<String>,

    #[arg(long, default_value = "104857600", env = "PARAKEET_MAX_UPLOAD_BYTES")]
    max_upload_bytes: usize,

    /// Maximum chunk duration in seconds for CTC/TDT models (prevents WebGPU buffer overflow).
    #[arg(long, default_value = "10", env = "PARAKEET_CHUNK_DURATION")]
    chunk_duration: u64,

    /// Execution provider for ONNX runtime.
    #[arg(long, value_enum, default_value = "webgpu", env = "PARAKEET_PROVIDER")]
    provider: Provider,
}

// ─── Model abstraction ────────────────────────────────────────────────────────

pub struct Transcript {
    pub text: String,
    pub tokens: Vec<Token>,
}

pub struct Token {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

enum Model {
    Ctc(Parakeet),
    Tdt(ParakeetTDT),
    Eou(ParakeetEOU),
    Nemotron(Nemotron),
}

impl Model {
    fn id(&self) -> &'static str {
        match self {
            Model::Ctc(_) => "parakeet-ctc-0.6b",
            Model::Tdt(_) => "parakeet-tdt-0.6b-v3",
            Model::Eou(_) => "parakeet-eou-120m-v1",
            Model::Nemotron(_) => "nemotron-streaming-0.6b",
        }
    }

    fn run(&mut self, audio: audio::AudioData, want_words: bool, chunk_duration: u64) -> anyhow::Result<Transcript> {
        match self {
            // ── CTC (chunked) ─────────────────────────────────────────────
            Model::Ctc(m) => {
                let mode = if want_words {
                    Some(TimestampMode::Words)
                } else {
                    Some(TimestampMode::Sentences)
                };
                Self::transcribe_chunked(m, audio, mode, chunk_duration)
            }

            // ── TDT (chunked) ─────────────────────────────────────────────
            Model::Tdt(m) => {
                let mode = if want_words {
                    Some(TimestampMode::Words)
                } else {
                    Some(TimestampMode::Sentences)
                };
                Self::transcribe_chunked(m, audio, mode, chunk_duration)
            }

            // ── EOU streaming ────────────────────────────────────────────────
            Model::Eou(m) => {
                let mono_16k = audio.to_16k_mono();
                const CHUNK: usize = 2560;
                let mut text = String::new();
                let chunks: Vec<&[f32]> = mono_16k.chunks(CHUNK).collect();
                let last = chunks.len().saturating_sub(1);
                for (i, chunk) in chunks.into_iter().enumerate() {
                    text.push_str(&m.transcribe(chunk, i == last)?);
                }
                Ok(Transcript {
                    text: text.trim().to_owned(),
                   tokens: vec![],
                })
            }

            // ── Nemotron streaming ────────────────────────────────────────────
            Model::Nemotron(m) => {
                let mono_16k = audio.to_16k_mono();
                const CHUNK: usize = 8960;
                let mut text = String::new();
                for chunk in mono_16k.chunks(CHUNK) {
                    text.push_str(&m.transcribe_chunk(chunk)?);
                }
                Ok(Transcript {
                    text: text.trim().to_owned(),
                   tokens: vec![],
                })
            }
        }
    }

    /// Chunk audio into segments of `chunk_duration` seconds and transcribe each.
    /// This prevents WebGPU buffer overflows on long files.
    fn transcribe_chunked<T: Transcriber>(
        model: &mut T,
        audio: audio::AudioData,
        mode: Option<TimestampMode>,
        chunk_duration: u64,
    ) -> anyhow::Result<Transcript> {
        let chunk_samples = (audio.sample_rate as u64 * audio.channels as u64 * chunk_duration) as usize;

        // If audio fits in one chunk, process directly
        if audio.samples.len() <= chunk_samples {
            let r = model.transcribe_samples(audio.samples, audio.sample_rate, audio.channels as u16, mode)?;
            return Ok(Transcript {
                text: r.text,
                tokens: r.tokens.into_iter().map(|t| Token {
                    text: t.text,
                    start: t.start as f64,
                    end: t.end as f64,
                }).collect(),
            });
        }

        // Process in chunks
        let mut full_text = String::new();
        let mut all_tokens: Vec<Token> = Vec::new();

        for (i, chunk) in audio.samples.chunks(chunk_samples).enumerate() {
            let chunk_offset_secs = (i * chunk_samples) as f64 / (audio.sample_rate as f64 * audio.channels as f64);

            let r = model.transcribe_samples(
                chunk.to_vec(),
                                             audio.sample_rate,
                                             audio.channels as u16,
                                             mode,
            )?;

            if !full_text.is_empty() && !r.text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(&r.text);

            // Offset timestamps so they're relative to the original file
            for t in r.tokens {
                all_tokens.push(Token {
                    text: t.text,
                    start: t.start as f64 + chunk_offset_secs,
                    end: t.end as f64 + chunk_offset_secs,
                });
            }
        }

        Ok(Transcript {
            text: full_text,
            tokens: all_tokens,
        })
    }
}

// ─── Shared state ─────────────────────────────────────────────────────────────

struct AppState {
    model: Mutex<Model>,
    api_key: Option<String>,
    max_duration: u64,
    chunk_duration: u64,
}

// ─── OpenAI-compatible response types ────────────────────────────────────────

#[derive(Serialize)]
struct TranscriptionJson {
    text: String,
}

#[derive(Serialize)]
struct VerboseJson {
    task: String,
    language: String,
    duration: f64,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    words: Option<Vec<WordTs>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    segments: Option<Vec<Segment>>,
}

#[derive(Serialize)]
struct WordTs {
    word: String,
    start: f64,
    end: f64,
}

#[derive(Serialize)]
struct Segment {
    id: usize,
    seek: u64,
    start: f64,
    end: f64,
    text: String,
    tokens: Vec<u32>,
    temperature: f32,
    avg_logprob: f32,
    compression_ratio: f32,
    no_speech_prob: f32,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

// ─── Handlers ────────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let id = {
        let m = state.model.lock().unwrap();
        m.id().to_owned()
    };
    Json(ModelList {
        object: "list".into(),
         data: vec![ModelInfo {
             id,
             object: "model".into(),
         created: 1_700_000_000,
         owned_by: "parakeet-rs".into(),
         }],
    })
}

async fn transcribe(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
                    mut multipart: Multipart,
) -> Result<Response, AppError> {
    if let Some(ref key) = state.api_key {
        let provided = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
        if provided != key {
            return Err(AppError::unauthorized("Invalid API key"));
        }
    }

    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut audio_name: Option<String> = None;
    let mut response_format = "json".to_owned();
    let mut timestamp_granularities: Vec<String> = vec![];

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(e.to_string()))?
        {
            let name = field.name().unwrap_or("").to_owned();
            match name.as_str() {
                "file" => {
                    audio_name = field.file_name().map(str::to_owned);
                    let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| AppError::bad_request(e.to_string()))?;
                    audio_bytes = Some(bytes.to_vec());
                }
                "response_format" => {
                    let b = field.bytes().await.map_err(|e| AppError::bad_request(e.to_string()))?;
                    response_format = String::from_utf8_lossy(&b).trim().to_owned();
                }
                "timestamp_granularities[]" | "timestamp_granularities" => {
                    let b = field.bytes().await.map_err(|e| AppError::bad_request(e.to_string()))?;
                    timestamp_granularities.push(String::from_utf8_lossy(&b).trim().to_owned());
                }
                _ => { let _ = field.bytes().await; }
            }
        }

        let audio_bytes = audio_bytes.ok_or_else(|| AppError::bad_request("Missing required field: file"))?;

        let ext = audio_name
        .as_deref()
        .and_then(|n| std::path::Path::new(n).extension())
        .and_then(|e| e.to_str())
        .unwrap_or("wav");

        let tmp = tempfile::Builder::new()
        .suffix(&format!(".{ext}"))
        .tempfile()
        .map_err(AppError::internal)?;

        std::fs::write(tmp.path(), &audio_bytes).map_err(AppError::internal)?;

        let audio_data = crate::audio::decode_audio(tmp.path())
        .map_err(|e| AppError::bad_request(format!("Audio decode error: {e}")))?;

        let duration_secs = audio_data.duration_secs();

        if state.max_duration > 0 && duration_secs > state.max_duration as f64 {
            return Err(AppError::payload_too_large(format!(
                "Audio duration {:.1}s exceeds server limit of {}s",
                duration_secs, state.max_duration
            )));
        }

        let want_words = timestamp_granularities.contains(&"word".to_owned())
        || response_format == "verbose_json";

        let chunk_duration = state.chunk_duration;
        let state2 = Arc::clone(&state);
        let transcript = tokio::task::spawn_blocking(move || {
            let mut m = state2.model.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
            m.run(audio_data, want_words, chunk_duration)
        })
        .await
        .map_err(AppError::internal)?
        .map_err(AppError::internal)?;

        let response: Response = match response_format.as_str() {
            "text" => axum::response::Response::builder()
            .header("Content-Type", "text/plain; charset=utf-8")
            .body(axum::body::Body::from(transcript.text))
            .unwrap(),

            "verbose_json" => {
                let words: Option<Vec<WordTs>> = if !transcript.tokens.is_empty() {
                    Some(transcript.tokens.iter().map(|t| WordTs {
                        word: t.text.clone(),
                                                      start: t.start,
                                                      end: t.end,
                    }).collect())
                } else {
                    None
                };

                let seg_start = transcript.tokens.first().map(|t| t.start).unwrap_or(0.0);
                let seg_end = transcript.tokens.last().map(|t| t.end).unwrap_or(duration_secs);

                let segments = Some(vec![Segment {
                    id: 0, seek: 0, start: seg_start, end: seg_end,
                    text: transcript.text.clone(), tokens: vec![],
                                    temperature: 0.0, avg_logprob: 0.0, compression_ratio: 1.0, no_speech_prob: 0.0,
                }]);

                Json(VerboseJson {
                    task: "transcribe".into(),
                     language: "en".into(),
                     duration: duration_secs,
                     text: transcript.text,
                     words, segments,
                }).into_response()
            }

            _ => Json(TranscriptionJson { text: transcript.text }).into_response(),
        };

        Ok(response)
}

// ─── Entry point ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
    .with_env_filter(
        tracing_subscriber::EnvFilter::from_default_env()
        .add_directive("parakeet_server=info".parse()?),
    )
    .init();

    let args = Args::parse();

    info!("Loading {:?} model from '{}'…", args.model_type, args.model_dir);

    let model_config = ExecutionConfig::new()
    .with_execution_provider(match args.provider {
        Provider::Cpu => ExecutionProvider::Cpu,
        Provider::Webgpu => ExecutionProvider::WebGPU,
    });

    let model = match args.model_type {
        ModelType::Ctc => Model::Ctc(
            Parakeet::from_pretrained(&args.model_dir, Some(model_config.clone()))
            .map_err(|e| anyhow::anyhow!("CTC load failed: {e}"))?,
        ),
        ModelType::Tdt => Model::Tdt(
            ParakeetTDT::from_pretrained(&args.model_dir, Some(model_config.clone()))
            .map_err(|e| anyhow::anyhow!("TDT load failed: {e}"))?,
        ),
        ModelType::Eou => Model::Eou(
            ParakeetEOU::from_pretrained(&args.model_dir, Some(model_config.clone()))
            .map_err(|e| anyhow::anyhow!("EOU load failed: {e}"))?,
        ),
        ModelType::Nemotron => Model::Nemotron(
            Nemotron::from_pretrained(&args.model_dir, Some(model_config))
            .map_err(|e| anyhow::anyhow!("Nemotron load failed: {e}"))?,
        ),
    };

    let model_id = model.id();
    info!("Loaded model: {model_id}");

    if let Some(ref key) = args.api_key {
        let redacted = format!("{}…{}", &key[..4.min(key.len())], &key[key.len().saturating_sub(4)..]);
        info!("API key auth enabled (key: {redacted})");
    }

    let state = Arc::new(AppState {
        model: Mutex::new(model),
                         api_key: args.api_key,
                         max_duration: args.max_duration,
                         chunk_duration: args.chunk_duration,
    });

    let cors = CorsLayer::new()
    .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
    .allow_headers(Any)
    .allow_origin(Any);

    let app = Router::new()
    .route("/v1/audio/transcriptions", post(transcribe))
    .route("/v1/models", get(list_models))
    .route("/health", get(health))
    .layer(DefaultBodyLimit::max(args.max_upload_bytes))
    .layer(cors)
    .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!("Server ready → http://{addr}");
    info!("  POST /v1/audio/transcriptions");
    info!("  GET  /v1/models");
    info!("  GET  /health");

    axum::serve(listener, app).await?;
    Ok(())
}
