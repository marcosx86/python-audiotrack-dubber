Your three problems are actually connected.

The biggest issue is not PyTorch performance. It's the timeline model used by the script.

Looking at your `speech_synthesis.py`, the critical line is:

```python
generated_dur = tts_speech.shape[1] / target_sr
current_time += generated_dur
```

Instead of respecting the original timestamps, you're advancing the timeline according to generated audio length. 

That means:

```text
Original:
0s ----- 3s

Generated:
0s -------- 5s
```

The next segment starts at 5s instead of 3s.

Repeat this for hundreds of segments and:

```text
Original video:
21 min

Generated audio:
33 min
```

which is exactly what you're seeing.

---

## Problem #1: Translation footprint

This is normal.

Example:

```text
EN:
You are born in Rome.

PT-BR:
Você nasceu em Roma.
```

Portuguese typically expands 15–40%.

German can expand even more.

Japanese often contracts.

No TTS model will fix this.

---

## Problem #2: Segment expansion

Right now you're doing:

```text
translate
   ↓
generate
   ↓
accept duration
```

For dubbing, you need:

```text
translate
   ↓
target duration known
   ↓
adapt text
   ↓
generate
```

The target duration already exists:

```python
target_duration = end - start
```

from your transcript. 

---

## What YouTube dubbing systems do

They don't directly translate.

Instead they perform:

```text
ASR
 ↓
Translation
 ↓
Condensation
 ↓
TTS
```

Example:

```text
EN:
I would like to explain to you how this works.

PT Literal:
Eu gostaria de explicar para você como isso funciona.

PT Dub:
Vou explicar como isso funciona.
```

Same meaning.

Much shorter.

Fits timing.

---

## What I would add

A duration-aware rewrite step.

Pseudo:

```python
target_duration = segment["duration"]

translated = translator(...)

adapted = llm(
    translated,
    target_duration=target_duration
)
```

Prompt:

```text
Rewrite this sentence in Brazilian Portuguese.

Requirements:
- preserve meaning
- preserve tone
- fit inside 3.2 seconds of speech
- shorten if necessary
```

This alone may reduce your 33 minutes to 24–26 minutes.

---

## Problem #3: CosyVoice inference throughput

Your Antigravity analysis is mostly correct.

You're seeing:

```text
GPU 59%
CPU 25%
```

because:

```python
for segment:
    inference_zero_shot(...)
```

is executed once per segment. 

Each iteration incurs:

```text
Python
 ↓
Torch
 ↓
CUDA launch
 ↓
GPU
 ↓
Python
```

hundreds or thousands of times.

---

## The bigger bottleneck

Ironically, this is probably not the largest slowdown.

This part is more expensive:

```python
extract_reference_audio(...)
```

for every segment. 

Your script:

```text
FFmpeg
 ↓
temporary wav
 ↓
CosyVoice
 ↓
delete wav
```

for every subtitle.

That's massive I/O churn.

---

## I would completely change reference handling

You already generated:

```text
reference.wav
reference.txt
```

Use them once.

Instead of:

```python
prompt_wav = segment_audio
prompt_text = orig_text
```

do:

```python
prompt_wav = reference.wav
prompt_text = reference.txt
```

for all segments.

Then remove:

```python
audio_extractor_worker()
extract_reference_audio()
threading queue
temp files
```

entirely.

That may double throughput immediately.

---

## Another issue

Current code performs:

```python
final_audio_pieces.append(tts_speech.cpu())
```

for every segment. 

This forces:

```text
GPU → CPU
```

transfer thousands of times.

That's expensive.

A better approach:

```text
generate segment
 ↓
save segment.wav
```

then assemble everything afterward.

Disk I/O is often cheaper than continuous GPU memory synchronization.

---

## What I would do next

Pipeline v2:

```text
WhisperX
      ↓
Translation
      ↓
Duration-aware rewrite
      ↓
CosyVoice
      ↓
segment_0001.wav
segment_0002.wav
...
      ↓
ffprobe duration
      ↓
light time-stretch
      ↓
timeline builder
      ↓
translated.wav
      ↓
ffmpeg mux
```

Most importantly:

**Stop using generated duration to advance the timeline.**

Use:

```python
current_time = end
```

or build the final track strictly from transcript timestamps.

The timeline should be dictated by the original transcript, not by CosyVoice output length. Once you make that architectural change, you'll be able to attack the remaining duration mismatch with text condensation and moderate time-stretching rather than allowing the entire dubbing track to drift by 12 minutes.
