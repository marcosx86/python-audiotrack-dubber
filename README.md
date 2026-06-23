# WhisperX Voice Extractor Pipeline

This project provides a robust, two-step pipeline for extracting high-quality audio segments of a specific speaker from an audio or video file. It is specifically designed to generate reference audio and text datasets tailored for Text-to-Speech (TTS) training or voice cloning.

## Prerequisites

- **Python 3.12** (Recommended, inside a **Conda environment**)
- **CUDA 12.6** (or compatible version)
- **PyTorch & TorchCodec**: Ensure you install the versions compatible with your CUDA version (e.g., `cu126`).
- **FFmpeg**: Requires a full **shared build** of FFmpeg (e.g., `v7.0.2-full_build-shared` on Windows). 
  - *Note:* Static builds might cause `libtorchcodec` loading errors on Windows Python 3.8+ due to missing DLLs.

### Key Python Packages

To successfully replicate this exact pipeline, the following major dependencies were configured in our Conda environment:

- **Audio Processing & TTS**: `torch`, `torchaudio`, `torchcodec`, `cosyvoice`
- **Transcription & Translation**: `whisperx`, `ctranslate2`
- **Local LLM Integration**: `openai` (used as a lightweight client for LM Studio / llama.cpp API endpoints)
- **Text Normalization (Windows)**: 
  - `pynini=2.1.6.post1` (Specifically requires Conda for pre-compiled Windows binaries)
  - `nemo_text_processing` (For advanced NeMo text normalization like `Normalizer('pt')`)

## Workflow

The pipeline consists of five main scripts:

### 1. `transcribe.py`

Transcribes the input media file, aligns the transcription, and performs speaker diarization using `WhisperX`.

**Key Features:**
- Uses `argparse` for easy parametrization.
- Robust logging (`INFO` and `DEBUG` via `--verbose`).
- **Low VRAM Support**: Includes a `--low-mem` option that explicitly deletes models and clears the CUDA cache (`gc.collect()`, `torch.cuda.empty_cache()`) between the Transcription, Alignment, and Diarization steps to prevent Out-Of-Memory (OOM) errors on consumer GPUs.

**Usage Example:**
```bash
python transcribe.py --low-mem --verbose
```
*(Check `python transcribe.py --help` for a full list of arguments and usage.)*

### 2. `extract_segments.py`

Reads the generated transcription text file, isolates segments spoken by a target speaker, and uses FFmpeg to extract and concatenate these clips.

**Key Features:**
- **Fast Extraction**: Uses FFmpeg input-seeking (`-ss` before `-i`) to prevent decoding the entire file, speeding up extraction significantly.
- **Parametrized Audio Processing**: Configure output sample rate, channels, and minimum segment duration (to filter out short "uh-huh" noises).
- **TTS Ready**: Generates a final concatenated `narrator_reference.wav`.
- **Text Companion**: Automatically generates `narrator_reference.txt` containing 0-indexed relative timestamps and the exact spoken text for each extracted clip in the concatenated file.

**Usage Example:**
```bash
python extract_segments.py transcript.txt media.mp4 --target-speaker SPEAKER_01 --min-duration 6.0
```
*(Check `python extract_segments.py --help` for a full list of arguments.)*

## Output Artifacts

Running the pipeline will generate several assets across its steps:

- `segments/`: A folder containing all the individual `.wav` clips extracted from the source (from `extract_segments.py`).
- `concat.txt`: The FFmpeg concatenation file list.
- `narrator_reference.wav`: The final concatenated audio file containing only the target speaker.
- `narrator_reference.txt`: A companion text file mapping the concatenated audio's new continuous timestamps to the spoken text.
- `<transcript_name>_<target_lang>.txt`: The fully translated transcription file keeping original timestamps intact.
- `translated_output.wav`: The final, fully synthesized translated audio track matching the absolute timeline of the original video.

### 3. `translate_transcription.py`

Translates the text of a WhisperX transcription file into a target language while preserving the timeline timestamps and speaker tags perfectly intact.

**Key Features:**
- **Local AI Engine**: Uses `CTranslate2` models and `AutoTokenizer` for offline, fast, and VRAM-efficient batch translation.
- **Language Query**: Includes a `--list-languages` parameter for easily querying common FLORES-200 target languages in the terminal.
- **Timeline Preservation**: Separates the timestamp (`[0.00s - 1.80s]`) prefix from text so translations drop perfectly back into the original timeline.

**Usage Example:**
```bash
python translate_transcription.py transcript.txt --model-dir path/to/nllb-model --target-lang por_Latn --compute-type int8
```
*(Check `python translate_transcription.py --help` for a full list of arguments.)*

### 4. `condense_transcription.py`

Condenses translated transcription lines using a local LLM via an OpenAI-compatible endpoint (like LM Studio). This is a crucial preprocessing step for dubbing languages that naturally expand significantly (like Portuguese from English), ensuring the synthesized audio can comfortably fit the original time bounds without requiring extreme, robotic time-stretching.

**Key Features:**
- **Local AI Pipeline**: Connects safely to local LLMs (LM Studio, llama.cpp) via the standard OpenAI API structure.
- **Dynamic Character Limiting**: Automatically calculates a strict target character limit per segment based on its specific duration in the timeline.
- **Hardware Protection**: Provides a `--cooldown` parameter to enforce delays between API calls and a `--context-length` parameter injected via `extra_body` during the JIT load warmup to prevent GPU overheating, BSODs, and VRAM OOMs during rapid, sequential inferences.
- **Smart Preservation**: Safely skips short sentences, gracefully buffers multiline transcriptions, flawlessly preserves the strict WhisperX prefix formatting, and protects against token hallucination via system prompt enforcement.

**Usage Example:**
```bash
python condense_transcription.py --output-file condensed_translation.txt --model qwen2.5-7b-instruct-1m@q4_k_m --temperature 0.2 --cooldown 1.5 --context-length 2048 translated_transcription.txt
```
*(Check `python condense_transcription.py --help` for a full list of arguments.)*

### 5. `speech_synthesis.py`

Synthesizes the translated text back into audio using the `CosyVoice` zero-shot TTS engine, aiming to emulate the pacing and tone of the original speaker, matched seamlessly to the original timestamps.

**Key Features:**
- **Asynchronous Extraction**: Implements a Producer-Consumer threading architecture with a managed queue, allowing CPU-bound FFmpeg audio slicing to pre-fetch perfectly in parallel with GPU-bound LLM generation.
- **Strict Timeline & Time-Stretching**: Enforces absolute synchronization to the original video by dynamically applying FFmpeg's `atempo` time-stretch filter to squeeze generated audio if it exceeds its original timestamp window.
- **Robust Text Normalization**: Intercepts numbers and normalizes them into localized Portuguese words using NVIDIA's `nemo_text_processing`, safely bypassing CosyVoice's default English normalization (`wetext`) for unsupported languages.
- **Hardware Optimized**: Includes `--fp16` support to slash memory bandwidth overhead and maximize GPU utilization during standard PyTorch autoregressive inference loops.
- **AutoModel Integration**: Natively supports `CosyVoice`, `CosyVoice2`, and `CosyVoice3` via dynamic factory initialization.
- **DLL Auto-Fixer**: Automatically registers your global `ffmpeg.exe` directory to cleanly bypass `torchcodec` load errors on Windows Python 3.8+.

**Usage Example:**
```bash
python .\speech_synthesis.py --ffmpeg-path C:/PATH_TO_FFMPEG_SHARED/bin/ffmpeg.exe --model-dir /COSYVOICE_MODELS/CosyVoice2-0.5B ORIGINAL_AUDIO.mp4 ORIGINAL_TRANSCRIPTION.txt TRANSLATED_TRANSCRIPTION.txt
```
*(Check `python speech_synthesis.py --help` for a full list of arguments.)*

Running `extract_segments.py` will create an output directory (default: `voice_extract`) containing:
- `segments/`: A folder containing all the individual `.wav` clips extracted from the source.
- `concat.txt`: The FFmpeg concatenation file list.
- `narrator_reference.wav`: The final concatenated audio file containing only the target speaker.
- `narrator_reference.txt`: A companion text file mapping the concatenated audio's new continuous timestamps to the spoken text. Example:
  ```text
  [0.00s - 13.47s] Unknown to you, somewhere across the city...
  [13.47s - 19.99s] They must also have no physical defects...
  ```

## Troubleshooting

- **`libtorchcodec` DLL Errors (Windows)**: If you encounter errors loading `torchcodec`, ensure you are using a shared build of FFmpeg. Python 3.8+ on Windows handles DLL paths strictly, so ensure the FFmpeg `bin` folder is correctly set in your environment.
- **CUDA OOM (Out of Memory) Errors**: When running the transcription, ensure you use the `--low-mem` flag to aggressively clean up VRAM between pipeline stages.
