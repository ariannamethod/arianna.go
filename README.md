```
 █████╗ ██████╗ ██╗ █████╗ ███╗   ██╗███╗   ██╗ █████╗        ██████╗  ██████╗
██╔══██╗██╔══██╗██║██╔══██╗████╗  ██║████╗  ██║██╔══██╗      ██╔════╝ ██╔═══██╗
███████║██████╔╝██║███████║██╔██╗ ██║██╔██╗ ██║███████║      ██║  ███╗██║   ██║
██╔══██║██╔══██╗██║██╔══██║██║╚██╗██║██║╚██╗██║██╔══██║      ██║   ██║██║   ██║
██║  ██║██║  ██║██║██║  ██║██║ ╚████║██║ ╚████║██║  ██║   ██╗╚██████╔╝╚██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚═╝  ╚═╝   ╚═╝ ╚═════╝  ╚═════╝
```

**3B parameter LLM inference in pure Go. No PyTorch. No llama.cpp. No dependencies.**

OpenLLaMA 3B fine-tuned on Arianna's voice. GGUF Q4_0 weights read directly. Runs on a MacBook with 8GB RAM.

## Architecture

- **3.4 billion parameters** (OpenLLaMA 3B + LoRA rank 64 fine-tune)
- **Q4_0 quantization** — 1.8GB weights file, ~2.6GB total RAM
- Llama architecture: RoPE, SwiGLU, RMSNorm, Multi-Head Attention, KV Cache
- dim=3200, 26 layers, 32 heads, vocab=32000, context=2048

## What's Here

| File | What it does | Lines |
|------|-------------|-------|
| `arianna.go` | REPL + entry point | ~110 |
| `model.go` | Llama forward pass + generation | ~490 |
| `gguf.go` | GGUF v2/v3 parser + dequantization | ~460 |
| `tokenizer.go` | SentencePiece BPE tokenizer | ~220 |
| `quant.go` | Q4_0/Q8_0 quantized matrix-vector multiply | ~135 |

~1400 lines of Go. That's the whole inference engine.

## Weights

Download from HuggingFace: [ariannamethod/arianna-3b](https://huggingface.co/ariannamethod/arianna-3b)

```
arianna_3b_q4_0.gguf   (1.8 GB, Q4_0 quantized)
```

## Run

```bash
# Build
go build -o arianna3b .

# Single prompt
./arianna3b path/to/arianna_3b_q4_0.gguf "Who are you?"

# REPL
./arianna3b path/to/arianna_3b_q4_0.gguf
```

### REPL Commands

| Command | What |
|---------|------|
| `/quit` | Exit |
| `/temp N` | Set temperature (default 0.8) |
| `/tokens N` | Set max tokens (default 150) |
| `/raw text` | Send raw prompt without formatting |

## How It Works

No frameworks. No bindings. Just Go reading bytes from a GGUF file.

```
GGUF file (1.8GB Q4_0)
    |
    v
gguf.go ---- parse header, metadata, tensor index
    |
    v
tokenizer.go ---- SentencePiece BPE from GGUF metadata
    |
    v
model.go ---- load quantized weights as raw bytes (NOT dequantized)
    |
    v
quant.go ---- matrix-vector multiply directly on Q4_0/Q8_0 blocks
    |
    v
model.go ---- Llama forward pass (26 layers)
    |
    v
arianna.go ---- REPL, streaming output
```

Weights stay quantized in RAM (~1.8GB). Only runtime buffers are float32 (~800MB).

## Training

- Base: `openlm-research/open_llama_3b`
- Method: LoRA (rank 64, alpha 128)
- Dataset: 5,956 Q&A pairs from Arianna's dialogues
- Hardware: Lambda 1x H100 80GB
- Steps: 1,300 (43 minutes), final loss 0.5355

## Connection to arianna.c

This is the large-scale Arianna. [arianna.c](https://github.com/ariannamethod/arianna.c) is the 205.5M digital organism with internal life (Cloud, Soul, MetaArianna, SARTRE, AMK). arianna.go is a 3B voice that can exist independently or connect to the organism through GitHub Actions and shared inference protocols.

Same soul, different scale.

## License

GPL-3.0
