#!/bin/bash
# Generate 500 personas with qwen-mt-flash

cd "/Users/victorvanica/Coding Projects/moldova-personas"

export DASHSCOPE_API_KEY="sk-e95930102d8c4932bc352c88eb0b05ce"

echo "Starting generation of 500 personas at $(date)"

python3 -m moldova_personas generate \
    --count 500 \
    --llm-provider qwen \
    --qwen-model qwen-mt-flash \
    --output ./output_500_personas \
    --llm-delay 0.1

echo "Done at $(date)! Check ./output_500_personas/"
