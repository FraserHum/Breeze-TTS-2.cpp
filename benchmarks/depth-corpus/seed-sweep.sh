P='The quick brown fox jumps over the lazy dog. We are benchmarking the Breeze TTS two model on the Radeon 780 M graphics card, measuring the real time factor of the full generation pipeline from the text encoder through the backbone and depth decoder.'
echo "SEED_SWEEP_START $(date -u +%H:%M:%S)"
for blocks in 9 12; do
  for seed in 0 1 7 123; do
    env GGML_VK_ALLOW_GRAPHICS_QUEUE=1 BREEZE_VOC_TRIM=1 BREEZE_VOC_CONVT_MATMUL=1 BREEZE_VOC_STATEFUL=0 BREEZE_DD_FUSED=0 BREEZE_DD_BLOCKS=$blocks \
      /src/build/breeze-cli /models/breeze-tts-2-q4_k.gguf \
      --text "$P" --instruction "Speak clearly and naturally." --seed $seed --repeat 1 --timings --output /tmp/sweep-$blocks-$seed.wav 2>&1 | \
      grep -E "frames, .* s audio|wall RTF" | sed "s/^/blocks=$blocks fused=0 seed=$seed /"
  done
done
echo "SEED_SWEEP_END $(date -u +%H:%M:%S)"
