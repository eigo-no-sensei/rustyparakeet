# parakeet-server

An **OpenAI-compatible ASR REST server** built on top of
[parakeet-rs](https://github.com/altunenes/parakeet-rs).  
Drop it in wherever you would use the OpenAI Whisper API.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/transcriptions` | Transcribe an audio file |
| `GET`  | `/v1/models`               | List the loaded model |
| `GET`  | `/health`                  | Liveness check |

The transcription endpoint accepts the same `multipart/form-data` fields
as the [OpenAI API](https://platform.openai.com/docs/api-reference/audio/createTranscription):

| Field | Type | Notes |
|-------|------|-------|
| `file` | file | **Required.** WAV, MP3, FLAC, OGG, AAC, AIFF |
| `model` | string | Accepted but ignored (server uses its loaded model) |
| `language` | string | Accepted; TDT auto-detects anyway |
| `response_format` | string | `json` (default) · `text` · `verbose_json` |
| `timestamp_granularities[]` | string | `word` · `segment` (CTC / TDT only) |
| `prompt` / `temperature` | — | Accepted and ignored |

---

## Model variants

| `--model-type` | Model | Languages | Timestamps |
|----------------|-------|-----------|------------|
| `ctc` (default) | Parakeet-CTC 0.6B | English | Word-level |
| `tdt` | Parakeet-TDT 0.6B v3 | 25 languages | Word / sentence |

---

## Setup

### 1 – Download model files

**CTC** (English, default): from [HuggingFace](https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/tree/main/onnx)
```
models/ctc/
  model.onnx
  model.onnx_data
  tokenizer.json
```

**TDT** (multilingual): from [HuggingFace](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
```
models/tdt/
  encoder-model.onnx
  encoder-model.onnx.data
  decoder_joint-model.onnx
  vocab.txt
```

**EOU** (streaming): from [HuggingFace](https://huggingface.co/altunenes/parakeet-rs/tree/main/realtime_eou_120m-v1-onnx)
```
models/eou/
  encoder.onnx
  decoder_joint.onnx
  tokenizer.json
```

**Nemotron** (streaming): from [HuggingFace](https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-speech-streaming-en-0.6b)
```
models/nemotron/
  encoder.onnx
  encoder.onnx.data
  decoder_joint.onnx
  tokenizer.model
```

### 2 – Build

```bash
cargo build --release
# GPU (pick one):
cargo build --release --features cuda
cargo build --release --features webgpu
```

### 3 – Run

```bash
# CTC (English, default)
./target/release/parakeet-server --model-dir ./models/ctc

# TDT (multilingual)
./target/release/parakeet-server --model-type tdt --model-dir ./models/tdt

# With auth key and custom port
./target/release/parakeet-server \
  --model-dir ./models/ctc \
  --port 8080 \
  --api-key my-secret-key
```

All options are also configurable via environment variables:

```bash
PARAKEET_MODEL_TYPE=tdt \
PARAKEET_MODEL_DIR=./models/tdt \
PARAKEET_PORT=9000 \
./target/release/parakeet-server
```

---

## Usage examples

### curl

```bash
# Plain JSON
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@speech.wav \
  -F model=whisper-1

# Verbose JSON with word timestamps
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@speech.mp3 \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=word"

# Plain text
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@speech.flac \
  -F response_format=text

# With auth
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer my-secret-key" \
  -F file=@speech.wav
```

### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",          # or your --api-key value
)

with open("speech.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",   # model name is ignored by the server
        file=f,
    )
print(result.text)
```

### Python (requests)

```python
import requests

with open("speech.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        files={"file": ("speech.wav", f, "audio/wav")},
    )
print(resp.json()["text"])
```

---

## Response formats

### `json` (default)
```json
{ "text": "Hello world." }
```

### `text`
```
Hello world.
```

### `verbose_json`
```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 3.14,
  "text": "Hello world.",
  "words": [
    { "word": "Hello", "start": 0.04, "end": 0.48 },
    { "word": "world.", "start": 0.52, "end": 1.10 }
  ],
  "segments": [
    {
      "id": 0, "seek": 0,
      "start": 0.04, "end": 1.10,
      "text": "Hello world.",
      "tokens": [], "temperature": 0.0,
      "avg_logprob": 0.0, "compression_ratio": 1.0, "no_speech_prob": 0.0
    }
  ]
}
```

> **Note:** `words` is only populated for CTC and TDT models when
> `timestamp_granularities[]=word` or `response_format=verbose_json` is
> requested. Streaming models (EOU, Nemotron) return empty `words`.

---

## License

MIT OR Apache-2.0 (same as parakeet-rs).
