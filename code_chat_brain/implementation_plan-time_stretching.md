# Pipeline v2 Architecture Review

I've reviewed the ChatGPT output, and it has absolutely nailed the structural flaws in our current approach, particularly regarding the **Timeline Drift**. Here is the breakdown of what is accurate, what is slightly misguided, and exactly how we should rebuild Pipeline v2.

## 1. The Timeline Drift (The Biggest Issue)
**ChatGPT's Point**: By using `current_time += generated_dur`, every translated sentence that runs longer than the original English sentence pushes the entire timeline forward. A 21-minute video becomes a 33-minute audio track, totally ruining the video sync.

**Our Fix**: We must enforce a **Strict Timeline Model**. 
- We will lock each segment to its original `start` and `end` timestamps.
- If the generated audio duration exceeds `(end - start)`, we will apply a **Time-Stretch** algorithm (using `torchaudio` SOX effects) to dynamically speed up that specific audio clip until it fits perfectly inside its original window. 

## 2. Reference Audio Churn
**ChatGPT's Point**: Slicing the video with FFmpeg to create a unique 3-second prompt for *every single sentence* is massive I/O churn. We should just use the `narrator_reference.wav` we generated earlier.

**Our Fix**: We don't need a Producer-Consumer thread or constant FFmpeg calls at all! `CosyVoice` only needs **one** good reference clip of the speaker to lock onto their voice.
- We will ask the user for a single `prompt.wav` and its corresponding `prompt.txt` (which you already generated via `extract_segments.py`).
- We load this *once* into RAM and pass it to `inference_zero_shot` for every segment. This immediately halves the complexity of the script and guarantees the GPU will run at maximum speed with zero I/O delays.

## 3. Memory Transfer Churn (ChatGPT is wrong here)
**ChatGPT's Point**: Calling `tts_speech.cpu()` thousands of times is expensive, so we should save to disk and assemble with FFmpeg.

**Reality Check**: `tts_speech.cpu()` takes 0.000 seconds. A 20-minute audio track takes less than 100MB of standard System RAM. Assembling the track perfectly in RAM and saving it *once* at the very end is infinitely faster and cleaner than writing 500 tiny `.wav` files to your hard drive and invoking FFmpeg. We will ignore this suggestion.

## Proposed Action Plan

1. **Remove the Producer Thread**: Strip out the queue and `audio_extractor_worker`.
2. **Add Universal Prompt Arguments**: Add `--prompt-wav` and `--prompt-text` arguments so we load one high-quality reference clip into RAM once.
3. **Implement Time-Stretching**: Introduce `torchaudio.sox_effects` to squeeze (`tempo`) the generated audio if it exceeds `(end - start)`.
4. **Strict Timeline Padding**: Revert the padding logic to strict absolute timestamps (`silence_dur = start - current_time`) and force `current_time = end` after stretching.

## Open Questions

> [!IMPORTANT]
> **To the User**: Do you agree with this Pipeline v2 approach? By using a single universal prompt, we can delete the complex threading logic. And by adding a `torchaudio` time-stretch pass, we guarantee your dubbed audio will perfectly match the original 21-minute video length without drifting!
