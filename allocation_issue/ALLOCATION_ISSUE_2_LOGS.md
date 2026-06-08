** Server Logs
127.0.0.1 - - [06/Jun/2026 12:00:05] "POST /process_image HTTP/1.1" 200 -
127.0.0.1 - - [06/Jun/2026 12:00:11] "GET /status/25197ac8-ccb7-4eb9-a6da-752de424c982 HTTP/1.1" 200 -
[TIMING] Preprocessing: 8.29s
 14%|█████████████████▍                                                                                                        | 10/70 [00:02<00:13,  4.51it/s]127.0.0.1 - - [06/Jun/2026 12:00:16] "GET /status/25197ac8-ccb7-4eb9-a6da-752de424c982 HTTP/1.1" 200 -
 50%|█████████████████████████████████████████████████████████████                                                             | 35/70 [00:07<00:07,  4.48it/s]127.0.0.1 - - [06/Jun/2026 12:00:21] "GET /status/25197ac8-ccb7-4eb9-a6da-752de424c982 HTTP/1.1" 200 -
 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████              | 62/70 [00:13<00:01,  4.50it/s]127.0.0.1 - - [06/Jun/2026 12:00:27] "GET /status/25197ac8-ccb7-4eb9-a6da-752de424c982 HTTP/1.1" 200 -
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 70/70 [00:15<00:00,  4.50it/s]
[TIMING] Multi-view diffusion (70 steps): 16.29s
[TIMING] GSLRM reconstruction: 1.09s
[TIMING] Save gaussians.ply: 1.54s
127.0.0.1 - - [06/Jun/2026 12:00:33] "GET /status/25197ac8-ccb7-4eb9-a6da-752de424c982 HTTP/1.1" 200 -


** Client Side Logs
useDemoFacelift.ts:69 [useDemoFacelift] Unexpected error: CUDA out of memory. Tried to allocate 130206.04 GiB. GPU 0 has a total capacity of 23.56 GiB of which 16.57 GiB is free. Including non-PyTorch memory, this process has 6.97 GiB memory in use. Of the allocated memory 5.45 GiB is allocated by PyTorch, and 1.17 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)