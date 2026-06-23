# Architecture Proposal: LLM Text Condensation

To fix the problem where Portuguese translations are naturally longer than the original English spoken words (which causes either timeline drift or chipmunk-speed time-stretching), we can introduce an **LLM Rewrite/Condensation** step.

Here is how we could optimally implement it in the pipeline:

## 1. Where in the Pipeline?
**Option A: A New Standalone Script (`condense_transcription.py`)**
We build a script that reads `translated_output.txt`. It looks at the `[start - end]` timestamps to calculate the maximum allowed time for each segment. It estimates if the translated text is too long (e.g., Portuguese averages ~15 characters spoken per second). If it's too long, it asks an LLM to rewrite it. It outputs a `condensed_output.txt`.
*Pros:* You get to manually review and edit the condensed text before spending time/VRAM generating the audio. 

**Option B: On-The-Fly inside `speech_synthesis.py`**
Right before NeMo normalization, we calculate `target_dur = end - start`. We estimate the required length. If the text is too long, the script pauses, pings an LLM to shorten it on the fly, and then immediately feeds the shorter text to CosyVoice.
*Pros:* Fully automated, no extra steps required.
*Cons:* If the LLM makes a mistake, it immediately gets burned into the audio.

## 2. The Condensation Logic
We would use a strict prompt to ensure the LLM maintains the meaning but reduces the syllable count:

> *"You are an expert dubbing scriptwriter. The following Portuguese text takes too long to speak. Rewrite it to be shorter, punchier, and more natural for spoken dialogue, keeping it strictly under X characters. Do not lose the core meaning."*

## 3. The Backend Engine
Because your current translator (`CTranslate2` / NLLB) is a pure translation model and *not* an instruction-following AI, we need an LLM backend for this step. We have two main options:

1. **OpenAI API (GPT-4o-mini)**: The easiest to code, incredibly fast, and very cheap. But it requires an active internet connection and an OpenAI API key.
2. **Local Ollama (Llama-3-8B)**: 100% offline and free. You would need to download the Ollama application to your Windows machine and pull a local model. The Python script would ping `localhost:11434`.

## Open Questions

> [!IMPORTANT]
> **To the User**: 
> 1. Do you prefer a **Standalone Script** (so you can review the text) or **On-The-Fly** (fully automated)?
> 2. Which backend do you want to use? **OpenAI API** or **Local Ollama**? Or maybe a HuggingFace cloud endpoint?
