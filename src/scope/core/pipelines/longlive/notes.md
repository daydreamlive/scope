longlive
480p
3 prompts
3 x 81 = 243 frames

# baseline

=== Performance Statistics ===
Latency - Avg: 1.13s, Max: 1.83s, Min: 0.95s
FPS - Avg: 10.79, Max: 12.62, Min: 6.56

# fp8 dynamic activation and weights quant

=== Performance Statistics ===
Latency - Avg: 2.45s, Max: 3.51s, Min: 2.30s
FPS - Avg: 4.91, Max: 5.21, Min: 3.38

# fp8 weights only quant

=== Performance Statistics ===
Latency - Avg: 1.27s, Max: 1.99s, Min: 1.08s
FPS - Avg: 9.53, Max: 11.08, Min: 6.03

# compile, max-autotune-no-cudagraphs, disabled self-attention forward pass

=== Performance Statistics ===
Latency - Avg: 1.36s, Max: 8.05s, Min: 0.84s
FPS - Avg: 11.50, Max: 14.30, Min: 1.49

# compile, max-autotune-no-cudagraphs, CacheIndices refactor (self-attn enabled)

=== Performance Statistics ===
Latency - Avg: 1.63s, Max: 13.24s, Min: 0.76s
FPS - Avg: 12.19, Max: 15.84, Min: 0.91

# i don't even know what claude did

=== Performance Statistics ===
Latency - Avg: 1.59s, Max: 16.33s, Min: 0.67s
FPS - Avg: 14.76, Max: 17.91, Min: 0.74

# the above but with lighttae

=== Performance Statistics ===
Latency - Avg: 1.25s, Max: 13.47s, Min: 0.43s
FPS - Avg: 20.17, Max: 27.69, Min: 0.89
