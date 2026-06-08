useDemoFacelift.ts:69 [useDemoFacelift] Unexpected error: CUDA out of memory. Tried to allocate 130212.62 GiB. GPU 0 has a total capacity of 23.56 GiB of which 16.92 GiB is free. Including non-PyTorch memory, this process has 6.62 GiB memory in use. Of the allocated memory 5.45 GiB is allocated by PyTorch, and 836.77 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
============================
ngrok tunnel URL: NgrokTunnel: "https://curtsy-halves-choice.ngrok-free.dev" -> "http://localhost:5001"
POST image to /process_image, then poll /status/<job_id> for results.
Processing 787f0121-b7b3-4e45-b13c-9a0cd4bde8c6.png
127.0.0.1 - - [06/Jun/2026 11:49:48] "POST /process_image HTTP/1.1" 200 -
127.0.0.1 - - [06/Jun/2026 11:49:54] "GET /status/787f0121-b7b3-4e45-b13c-9a0cd4bde8c6 HTTP/1.1" 200 -
[TIMING] Preprocessing: 10.48s
127.0.0.1 - - [06/Jun/2026 11:49:59] "GET /status/787f0121-b7b3-4e45-b13c-9a0cd4bde8c6 HTTP/1.1" 200 -
 29%|██████████████████████████████████▊                                                                                       | 20/70 [00:04<00:11,  4.49it/s]127.0.0.1 - - [06/Jun/2026 11:50:04] "GET /status/787f0121-b7b3-4e45-b13c-9a0cd4bde8c6 HTTP/1.1" 200 -
 63%|████████████████████████████████████████████████████████████████████████████▋                                             | 44/70 [00:10<00:05,  4.50it/s]127.0.0.1 - - [06/Jun/2026 11:50:10] "GET /status/787f0121-b7b3-4e45-b13c-9a0cd4bde8c6 HTTP/1.1" 200 -
 97%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌   | 68/70 [00:15<00:00,  4.49it/s]127.0.0.1 - - [06/Jun/2026 11:50:15] "GET /status/787f0121-b7b3-4e45-b13c-9a0cd4bde8c6 HTTP/1.1" 200 -
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 70/70 [00:15<00:00,  4.43it/s]
[TIMING] Multi-view diffusion (70 steps): 17.29s
[TIMING] GSLRM reconstruction: 1.30s
[TIMING] Save gaussians.ply: 1.62s
127.0.0.1 - - [06/Jun/2026 11:50:21] "GET /status/787f0121-b7b3-4e45-b13c-9a0cd4bde8c6 HTTP/1.1" 200 -