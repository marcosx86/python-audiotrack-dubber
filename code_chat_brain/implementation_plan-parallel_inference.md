# Parallel Inference Pipeline Implementation Plan

To achieve highly optimized GPU utilization using parallel inference threads without breaking chronological timeline merging, we will implement an Asynchronous Dispatcher with a Reorder Buffer using Python's `ThreadPoolExecutor`.

## The Architectural Design

### 1. Decoupling Inference from Post-Processing
Currently, the main loop handles both GPU inference and FFmpeg time-stretching as one giant sequential block. We will split this into two distinct operations:
- **`run_inference_task(...)`**: A standalone function that takes text, normalizes it via NeMo, extracts XTTS latents, runs `xtts.inference()`, and returns the raw PyTorch audio tensor. This function will be thread-safe and heavily GPU-bound.
- **Timeline Merging Loop**: A sequential operation in the Main Thread that takes the returned tensor, calculates silence padding (`current_time`), runs `apply_time_stretch_ffmpeg()` if needed, and appends it to `final_audio_pieces`.

### 2. The `ThreadPoolExecutor` Dispatcher
In `main()`, we will spin up a `ThreadPoolExecutor` with a configurable `max_workers` parameter (e.g., `--num-workers 2`). 

The Main Thread will continuously pull extracted reference audio from the FFmpeg `task_queue` and dispatch them into the thread pool as `Future` objects.

### 3. The Reorder Buffer
Because multiple inference threads run asynchronously, Segment 3 might finish before Segment 2. 
To guarantee chronological audio merging, the Main Thread will maintain a dictionary acting as a Reorder Buffer: `pending_futures = {}`.

**The Sliding Window Loop:**
As the Main Thread dispatches tasks, if the number of pending tasks hits a limit (e.g., `num_workers * 2`), the Main Thread will intentionally block and wait for `pending_futures[next_idx_to_process].result()`. 
Even if Segment 3 is finished, it forces the script to wait for Segment 2 to finish, processes Segment 2 into the timeline, and only *then* grabs the already-finished Segment 3. 

This flawlessly guarantees chronological ordering without complex PriorityQueues.

## Proposed Code Changes in `xttsv2_speech_synthesis.py`

### 1. Argparse Addition
```python
parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel XTTS inference threads. Warning: N>1 multiplies VRAM usage. Start with 2.")
```

### 2. Standalone Inference Function
```python
def run_inference_worker(task_idx, seg, reference_audio_path, extract_error, xtts, nemo_normalizer, args):
    # Handles: NeMo Normalization, Latent Extraction, xtts.inference(), CPU Tensor Conversion
    # Returns a dictionary: {'idx': task_idx, 'tensor': tts_speech, 'seg': seg, 'error': None}
```

### 3. Main Loop Refactor
```python
executor = ThreadPoolExecutor(max_workers=args.num_workers)
pending_futures = {}
next_idx_to_process = 0

# The Dispatch & Process Loop
while True:
    task = task_queue.get()
    
    if task is not None:
        # 1. Dispatch asynchronous inference
        future = executor.submit(run_inference_worker, task['idx'], task['seg'], ...)
        pending_futures[task['idx']] = future

    # 2. Reorder Buffer Consumer
    # We must process the timeline sequentially. We block and wait for the 'next' chronological
    # task if our buffer is getting full, or if we have finished dispatching everything (task is None).
    while len(pending_futures) >= args.num_workers * 2 or (task is None and next_idx_to_process < len(segments)):
        if next_idx_to_process in pending_futures:
            result = pending_futures.pop(next_idx_to_process).result()
            
            # --- START TIMELINE MODEL ---
            # Extract result['tensor'], calculate silence padding, run apply_time_stretch_ffmpeg, append to final_audio_pieces
            # ...
            # --- END TIMELINE MODEL ---
            
            next_idx_to_process += 1
        else:
            break
            
    if task is None:
        break # Everything is dispatched and flushed
```

## Open Questions for You
> [!WARNING]
> Coqui TTS's `xtts.inference()` function is known to have some global state locks under the hood in PyTorch depending on the version. Running multiple parallel forward passes *might* occasionally crash if they try to access the exact same memory tensor simultaneously without batching. 
> 
> The only way to know if your local CUDA driver handles it cleanly is to try it. Are you ready to proceed with this implementation and test it on your hardware?
