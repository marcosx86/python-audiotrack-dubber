# Architecture Proposal: LLM Text Condensation

## 1. Pipeline Location: Standalone Script
As requested, we will build a dedicated script: `condense_transcription.py`. 

**Workflow:**
1. You run `python condense_transcription.py original_audio.txt translated_audio.txt --max-chars-per-sec 15`
2. The script calculates the duration of each segment `[start - end]`.
3. It multiplies the duration by the target character rate (e.g., 15 chars/sec) to find the strict `target_length`.
4. If `len(translated_text)` exceeds `target_length`, it sends it to the LLM to be condensed.
5. It saves a new file: `condensed_translation.txt`
6. You can manually review this text file before passing it into `speech_synthesis.py`!

## 2. Backend Engine: LM Studio vs llama.cpp
I **highly recommend using LM Studio's OpenAI-compatible endpoint** over a native `llama.cpp` python implementation. Here is why:

1. **Compilation Nightmares**: Installing `llama-cpp-python` with CUDA acceleration on Windows often requires installing Visual Studio C++ build tools and CMake, which is notoriously frustrating and error-prone.
2. **VRAM Management**: If we embed the LLM directly into the script, Python has to constantly load and unload massive 5GB+ model weights into your GPU alongside your other scripts. LM Studio runs gracefully in the background and manages VRAM perfectly.
3. **Zero Dependencies**: By using LM Studio's endpoint, our python script stays incredibly lightweight. We just `pip install openai` and point it to `http://localhost:1234/v1`.

### Prompt Engineering (Avoiding Garbage)
Because local LLMs love to "chat" (e.g., *"Here is your condensed text: ..."*), we will use a rigorous System Prompt structure combined with the OpenAI API's `temperature=0.1` to force strict formatting:

**System Prompt:**
> "You are a professional dubbing scriptwriter. Your job is to rewrite Portuguese subtitles to be shorter, punchier, and faster to speak, without losing the core meaning. You must strictly obey the character limit provided. OUTPUT ONLY THE CONDENSED TEXT. Do not output conversational filler, introductions, quotes, or explanations."

**User Prompt:**
> "Condense the following text to be under X characters: '{original_text}'"

## User Review Required

> [!IMPORTANT]
> **To the User:**
> Do you approve this LM Studio API approach? If so, simply:
> 1. Run `pip install openai` in your Conda environment.
> 2. Open LM Studio, load your favorite model (e.g., Llama-3-8B-Instruct), and click **Start Server**.
> 3. Give me the green light to write `condense_transcription.py`!
