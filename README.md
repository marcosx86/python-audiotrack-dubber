# WhisperX Voice Extractor Pipeline

This project provides a robust, two-step pipeline for extracting high-quality audio segments of a specific speaker from an audio or video file. It is specifically designed to generate reference audio and text datasets tailored for Text-to-Speech (TTS) training or voice cloning.

## Prerequisites

- **Python 3.12** (Recommended, inside a virtual environment `.venv`)
- **CUDA 12.6** (or compatible version)
- **PyTorch & TorchCodec**: Ensure you install the versions compatible with your CUDA version (e.g., `cu126`).
- **FFmpeg**: Requires a full **shared build** of FFmpeg (e.g., `v7.0.2-full_build-shared` on Windows). 
  - *Note:* Static builds might cause `libtorchcodec` loading errors on Windows Python 3.8+ due to missing DLLs.

## Workflow

The pipeline consists of two main scripts:

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
