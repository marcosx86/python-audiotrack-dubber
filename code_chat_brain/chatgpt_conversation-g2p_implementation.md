Yes. In fact, if your target is Portuguese dubbing, I'd seriously consider replacing WeText entirely.

The WeText frontend in CosyVoice serves several purposes:

```text
Input text
    ↓
Normalization
    ↓
Sentence segmentation
    ↓
G2P (grapheme-to-phoneme)
    ↓
Prosody hints
    ↓
Acoustic model
```

The problem is exactly what you found:

```text
Official support:
- English
- Chinese
- Japanese
```

Portuguese ends up passing through paths that were never properly trained.

This can explain part of the "source-language footprint" you're hearing.

---

## Option 1 — Use Espeak-NG as G2P

This is probably the easiest path.

[eSpeak NG](https://github.com/espeak-ng/espeak-ng?utm_source=chatgpt.com)

supports:

```text
Portuguese (pt)
Portuguese BR (pt-br)
Spanish
French
German
Italian
...
```

You can generate phonemes:

```bash
espeak-ng -v pt-br --ipa "Você nasceu em Roma"
```

Output:

```text
voˈse naˈsew ẽ ˈʁomɐ
```

Advantages:

* Mature
* Offline
* Fast
* Hundreds of languages

---

## Option 2 — Piper Frontend

[Piper TTS](https://github.com/rhasspy/piper?utm_source=chatgpt.com)

Piper contains a surprisingly good multilingual frontend.

People have already extracted portions of Piper's normalization and G2P stack for other TTS projects.

For PT-BR pronunciation it tends to outperform eSpeak.

---

## Option 3 — Misaki G2P

This is currently one of the most interesting options.

[Misaki G2P](https://github.com/hexgrad/misaki?utm_source=chatgpt.com)

Supported languages include:

```text
English
Portuguese
Spanish
French
German
Italian
```

It was specifically created for modern neural TTS systems.

Advantages:

```text
Better prosody
Better punctuation handling
More natural pronunciation
```

than classic eSpeak.

If I were modifying CosyVoice today, this would be the first thing I'd test.

---

## Option 4 — Bypass WeText Completely

Depending on your CosyVoice version, you may be able to inject normalized text directly.

Instead of:

```python
cosyvoice.inference_zero_shot(
    tts_text=text,
    ...
)
```

you preprocess:

```python
text
  ↓
normalizer
  ↓
g2p
  ↓
phonemes
```

and feed phoneme sequences into the frontend encoder.

This requires modifying CosyVoice internals but gives maximum control.

---

## Option 5 — Use a Small LLM as Text Frontend

For dubbing, this is actually where I'd spend effort.

Current flow:

```text
Translation
      ↓
CosyVoice
```

Better:

```text
Translation
      ↓
LLM Rewriter
      ↓
CosyVoice
```

Example:

Literal translation:

```text
Eu gostaria de explicar para você como isso funciona.
```

Dubbing rewrite:

```text
Vou explicar como isso funciona.
```

This simultaneously:

* reduces duration
* improves naturalness
* removes translation artifacts
* improves TTS prosody

---

## What I'd investigate in your codebase

Before replacing WeText, I'd verify whether the language footprint is actually coming from the frontend.

CosyVoice's strongest cross-lingual path is usually:

```python
inference_cross_lingual(...)
```

rather than:

```python
inference_zero_shot(...)
```

Many users report:

```text
English voice
→ Portuguese speech

sounds more native
```

through the cross-lingual path.

If your current script is calling:

```python
cosyvoice.inference_zero_shot(
    tts_text=trans_text,
    prompt_text=orig_text,
    prompt_wav=reference_audio
)
```

then I'd test `inference_cross_lingual()` before rewriting the frontend. In many cases the accent issue is coming from the inference mode, not from WeText itself.

For your Portuguese dubbing pipeline, my order of experiments would be:

1. Try `inference_cross_lingual()`.
2. Add LLM-based duration-aware rewriting.
3. Replace WeText G2P with Misaki.
4. Only then consider deeper frontend surgery inside CosyVoice.

I suspect step 2 will have a larger impact on the final result than step 3, because the 21→33 minute expansion indicates the translation layer is currently contributing more quality loss than the phoneme frontend.
