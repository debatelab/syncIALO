#!bin/bash

MODEL_ID="TheBloke/SOLAR-10.7B-Instruct-v1.0-AWQ"
PORT="8899"
IP="0.0.0.0"
ENDPOINT="${IP}:${PORT}"

lmql serve-model $MODEL_ID \
  --batch_size 128 \
  --static \
  --use_safetensors true \
  --cuda \
  --host $IP \
  --port $PORT  &
PID=$!

sleep 30

#  --low_cpu_mem_usage true \
#  --dtype "float16" \
#  --attn_implementation "flash_attention_2" \


echo "PID lmql server: $PID"
