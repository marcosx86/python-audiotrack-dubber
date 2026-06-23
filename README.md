# WhisperX Voice Extractor Pipeline

This project provides a robust, two-step pipeline for extracting high-quality audio segments of a specific speaker from an audio or video file. It is specifically designed to generate reference audio and text datasets tailored for Text-to-Speech (TTS) training or voice cloning.

## Prerequisites

- **Python 3.12** (Recommended, inside a virtual environment `.venv`)
- **CUDA 12.6** (or compatible version)
- **PyTorch & TorchCodec**: Ensure you install the versions compatible with your CUDA version (e.g., `cu126`).
- **FFmpeg**: Requires a full **shared build** of FFmpeg (e.g., `v7.0.2-full_build-shared` on Windows). 
  - *Note:* Static builds might cause `libtorchcodec` loading errors on Windows Python 3.8+ due to missing DLLs.

## Workflow

The pipeline consists of four main scripts:

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

### 4. `speech_synthesis.py`

Synthesizes the translated text back into audio using the `CosyVoice` zero-shot TTS engine, aiming to emulate the pacing and tone of the original speaker, matched seamlessly to the original timestamps.

**Key Features:**
- **Dynamic Timeline Alignment**: Reads both original and translated transcriptions simultaneously. If the translation requires more or less time to speak, the script computes and pads exact silence gaps so the next sentence starts precisely at its original absolute timestamp.
- **RAM-Efficient Audio Slicing**: Uses `torchaudio` combined with an `ffmpeg` subprocess to slice only the required reference context directly from the full source file, converting it to a 16kHz prompt on the fly.
- **AutoModel Integration**: Natively supports `CosyVoice`, `CosyVoice2`, and `CosyVoice3` via dynamic factory initialization.
- **DLL Auto-Fixer**: Automatically detects your global `ffmpeg.exe` and securely registers its directory to bypass `torchcodec` DLL load errors on Windows.

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
