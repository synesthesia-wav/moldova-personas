# Qwen Setup Guide

This guide covers using Qwen (Alibaba's LLM) for generating narrative content in Moldovan personas.

## Why Qwen?

- **Excellent multilingual capabilities** - Strong Romanian language support
- **Cost-effective** - Often cheaper than GPT-3.5/4
- **Open weights** - Can run locally with privacy
- **Multiple sizes** - 0.5B to 110B parameters

## Option 1: Qwen via DashScope API (Recommended)

Fastest setup, pay-per-use pricing.

### Installation

```bash
pip install dashscope
```

### Get API Key

1. Sign up at [dashscope.aliyun.com](https://dashscope.aliyun.com)
2. Create an API key
3. Set environment variable:
   ```bash
   export DASHSCOPE_API_KEY="sk-..."
   ```

### Usage

```bash
# Using qwen-turbo (fastest, most economical)
python -m moldova_personas generate \
    --count 1000 \
    --llm-provider qwen \
    --qwen-model qwen-turbo \
    --output ./output

# Using qwen-plus (better quality)
python -m moldova_personas generate \
    --count 1000 \
    --llm-provider qwen \
    --qwen-model qwen-plus \
    --output ./output

# Using qwen-max (best quality)
python -m moldova_personas generate \
    --count 1000 \
    --llm-provider qwen \
    --qwen-model qwen-max \
    --output ./output

# Via demo script
python demo_narrative.py --mode qwen --count 5
```

### Available Models

| Model | Quality | Speed | Cost (per 1K personas) |
|-------|---------|-------|------------------------|
| `qwen-turbo` | Good | Fastest | ~$0.05 |
| `qwen-plus` | Better | Fast | ~$0.20 |
| `qwen-max` | Best | Normal | ~$1.00 |
| `qwen2.5-7b` | Good | Fast | ~$0.05 |
| `qwen2.5-14b` | Better | Normal | ~$0.10 |
| `qwen2.5-72b` | Best | Slower | ~$0.50 |

## Option 2: Local Qwen (Privacy, No API Costs)

Run Qwen on your own hardware. No internet required after download.

### Hardware Requirements

| Model | VRAM Required | RAM Required | Speed (per persona) |
|-------|---------------|--------------|---------------------|
| Qwen2.5-7B | 16 GB | 32 GB | ~5-10 seconds |
| Qwen2.5-7B (4-bit) | 8 GB | 16 GB | ~5-10 seconds |
| Qwen2.5-14B | 28 GB | 48 GB | ~10-20 seconds |
| Qwen2.5-14B (4-bit) | 12 GB | 24 GB | ~10-20 seconds |
| Qwen2.5-32B | 64 GB | 96 GB | ~20-40 seconds |

### Installation

```bash
# Install PyTorch (CUDA version if you have GPU)
pip install torch torchvision torchaudio

# Or CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install transformers and accelerate
pip install transformers>=4.35.0 accelerate

# Optional: For 4-bit quantization (saves VRAM)
pip install bitsandbytes
```

### Usage

```bash
# 7B model (most accessible)
python -m moldova_personas generate \
    --count 100 \
    --llm-provider qwen-local \
    --qwen-local-model qwen2.5-7b \
    --output ./output

# 14B model (better quality)
python -m moldova_personas generate \
    --count 100 \
    --llm-provider qwen-local \
    --qwen-local-model qwen2.5-14b \
    --output ./output

# Via demo script
python demo_narrative.py --mode qwen-local --count 3
```

### Python API

```python
from moldova_personas import PersonaGenerator
from moldova_personas.narrative_generator import NarrativeGenerator

# Generate structured data
generator = PersonaGenerator(seed=42)
personas = generator.generate(100)

# Add narratives with local Qwen
nar_gen = NarrativeGenerator(
    provider="qwen-local",
    model_name="qwen2.5-7b"  # or "Qwen/Qwen2.5-7B-Instruct"
)
personas = nar_gen.generate_batch(personas)
```

### GPU Acceleration

For faster generation with local models:

```python
# NVIDIA GPU with CUDA
nar_gen = NarrativeGenerator(
    provider="qwen-local",
    model_name="qwen2.5-7b",
    device="cuda"  # Auto-detects GPU
)

# Apple Silicon (M1/M2/M3)
nar_gen = NarrativeGenerator(
    provider="qwen-local",
    model_name="qwen2.5-7b",
    device="mps"  # Metal Performance Shaders
)
```

### 4-Bit Quantization

Run larger models with less VRAM:

```python
nar_gen = NarrativeGenerator(
    provider="qwen-local",
    model_name="qwen2.5-14b",
    load_in_4bit=True  # Reduces VRAM by ~75%
)
```

## Cost Comparison (100k Personas)

| Provider | Model | Cost | Time | Quality |
|----------|-------|------|------|---------|
| **DashScope** | qwen-turbo | ~$50 | ~1.5h | ⭐⭐⭐ |
| **DashScope** | qwen-plus | ~$200 | ~2h | ⭐⭐⭐⭐ |
| **DashScope** | qwen-max | ~$1000 | ~3h | ⭐⭐⭐⭐⭐ |
| **Local** | qwen2.5-7b | $0 | ~20-40h | ⭐⭐⭐ |
| **Local** | qwen2.5-14b | $0 | ~40-80h | ⭐⭐⭐⭐ |
| **OpenAI** | gpt-3.5-turbo | ~$500 | ~2h | ⭐⭐⭐ |
| **OpenAI** | gpt-4 | ~$5000 | ~2h | ⭐⭐⭐⭐⭐ |

## Testing

Test Qwen integration without generating full personas:

```bash
# Test DashScope API
python -c "
from moldova_personas.llm_client import create_llm_client
client = create_llm_client('qwen', model='qwen-turbo')
response = client.generate('Salut! Spune-mi ceva despre Moldova în română.')
print(response)
"

# Test local Qwen (downloads model on first run)
python -c "
from moldova_personas.llm_client import create_llm_client
client = create_llm_client('qwen-local', model_name='qwen2.5-7b')
response = client.generate('Salut! Spune-mi ceva despre Moldova în română.')
print(response)
"
```

## Troubleshooting

### "CUDA out of memory" for local models

Use 4-bit quantization:
```bash
python -m moldova_personas generate \
    --llm-provider qwen-local \
    --qwen-local-model qwen2.5-14b \
    # Add to the code:
    # load_in_4bit=True in llm_client.py
```

Or use a smaller model:
```bash
python -m moldova_personas generate \
    --llm-provider qwen-local \
    --qwen-local-model qwen2.5-7b
```

### DashScope API errors

Check your API key:
```bash
echo $DASHSCOPE_API_KEY
```

Verify it's set correctly in Python:
```python
import os
print(os.getenv("DASHSCOPE_API_KEY"))
```

### Slow generation with local models

- Use GPU if available (`device="cuda"`)
- Reduce batch size
- Use smaller model (7B vs 14B)
- Use qwen-turbo via API instead of local

## Recommended Workflow

1. **Test with mock mode**: Verify structured data generation
   ```bash
   python demo_narrative.py --mode mock --count 5
   ```

2. **Test with qwen-turbo**: Quick quality check
   ```bash
   python demo_narrative.py --mode qwen --count 5
   ```

3. **Generate sample (1k)**: Evaluate quality
   ```bash
   python -m moldova_personas generate --count 1000 --llm-provider qwen
   ```

4. **Full production run**: Scale to 100k
   ```bash
   python -m moldova_personas generate --count 100000 --llm-provider qwen --qwen-model qwen-plus
   ```

## Additional Resources

- [Qwen GitHub](https://github.com/QwenLM/Qwen)
- [Qwen HuggingFace](https://huggingface.co/Qwen)
- [DashScope Documentation](https://help.aliyun.com/document_detail/611472.html)
- [Qwen2.5 Paper](https://arxiv.org/abs/2409.12186)
