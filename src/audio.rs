//! Audio decoding: any format Symphonia supports → native-rate interleaved f32.
//!
//! For CTC and TDT models the library accepts any sample rate / channel count
//! and handles preprocessing internally, so we pass the native values through.
//!
//! For streaming models (EOU, Nemotron) the caller resamples to 16 kHz mono
//! before chunking; `AudioData::to_16k_mono()` is provided for that purpose.

use anyhow::{anyhow, Result};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{DecoderOptions, CODEC_TYPE_NULL},
    errors::Error as SymphError,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

/// Decoded audio in its native format.
pub struct AudioData {
    /// Interleaved f32 samples, normalised to [-1, 1].
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u32,
}

impl AudioData {
    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / (self.sample_rate as f64 * self.channels as f64)
    }

    /// Convert to 16 kHz mono — required by streaming models (EOU, Nemotron).
    pub fn to_16k_mono(&self) -> Vec<f32> {
        let mono = if self.channels == 1 {
            self.samples.clone()
        } else {
            self.samples
                .chunks(self.channels as usize)
                .map(|frame| frame.iter().sum::<f32>() / self.channels as f32)
                .collect()
        };

        if self.sample_rate == 16_000 {
            mono
        } else {
            linear_resample(mono, self.sample_rate as usize, 16_000)
        }
    }
}

/// Decode any audio file Symphonia can open.
/// Returns samples at native sample rate and channel layout.
/// CTC / TDT model variants handle resampling internally when you pass
/// the `sample_rate` and `channels` values from this struct.
pub fn decode_audio(path: &std::path::Path) -> Result<AudioData> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("No valid audio track found"))?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44_100);
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u32)
        .unwrap_or(1);

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphError::ResetRequired) => continue,
            Err(SymphError::IoError(_)) => break,
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let mut buf =
                    SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                buf.copy_interleaved_ref(decoded);
                samples.extend_from_slice(buf.samples());
            }
            Err(SymphError::DecodeError(e)) => {
                tracing::warn!("Skipping corrupt packet: {e}");
                continue;
            }
            Err(e) => return Err(e.into()),
        }
    }

    if samples.is_empty() {
        return Err(anyhow!("No audio samples decoded"));
    }

    tracing::debug!(
        "Decoded {} samples @ {} Hz × {} ch ({:.2}s)",
        samples.len(),
        sample_rate,
        channels,
        samples.len() as f64 / (sample_rate as f64 * channels as f64),
    );

    Ok(AudioData { samples, sample_rate, channels })
}

/// Linear interpolation resampler (mono).
pub fn linear_resample(input: Vec<f32>, from_hz: usize, to_hz: usize) -> Vec<f32> {
    if from_hz == to_hz {
        return input;
    }
    let ratio = from_hz as f64 / to_hz as f64;
    let out_len = ((input.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * ratio;
        let idx = src as usize;
        let frac = (src - idx as f64) as f32;
        let s0 = input.get(idx).copied().unwrap_or(0.0);
        let s1 = input.get(idx + 1).copied().unwrap_or(s0);
        out.push(s0 + (s1 - s0) * frac);
    }
    out
}