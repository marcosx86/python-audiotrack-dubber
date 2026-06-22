import os
import sys
import gc
import argparse
import logging

def parse_args():
    parser = argparse.ArgumentParser(
        description="WhisperX offline transcription, word-level alignment, and speaker diarization tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("audio_file", help="Path to the audio or video file to transcribe")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use for computation")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for transcription (reduce if low on VRAM)")
    parser.add_argument("--compute-type", default="float16", help="Compute type (change to 'int8' if low on VRAM)")
    parser.add_argument("--model-path", default="D:/Models/faster-whisper-large-v3", help="Path to local CTranslate2 faster-whisper model")
    parser.add_argument("--align-model", default="D:/Models/wav2vec2-large-xlsr-53-english", help="Path or Hugging Face ID of alignment model")
    parser.add_argument("--diarize-config", default="D:/Models/speaker-diarization-community-1/config.yaml", help="Path to speaker diarization config.yaml")
    parser.add_argument("--min-speakers", type=int, default=1, help="Minimum number of speakers to detect")
    parser.add_argument("--max-speakers", type=int, default=3, help="Maximum number of speakers to detect")
    parser.add_argument("--ffmpeg-dir", default=r"C:\Programas\ffmpeg-7.0.2-full_build-shared\bin", help="Path to FFmpeg shared DLLs bin folder")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--low-mem", action="store_true", help="Explicitly clean up and unload models from VRAM between each step")
    return parser.parse_args()

def setup_environment(ffmpeg_dir):
    # Register the FFmpeg shared DLL directory (for Windows)
    if sys.platform == "win32":
        if os.path.isdir(ffmpeg_dir):
            os.add_dll_directory(ffmpeg_dir)
            logging.info(f"Registered FFmpeg DLL directory: {ffmpeg_dir}")
        else:
            logging.warning(f"FFmpeg DLL directory not found: {ffmpeg_dir}. DLL load may fail.")

    # Set offline environment variables to prevent network connections
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - [%(levelname)s] - %(message)s")

    # Set up DLL paths and env variables before importing heavy packages
    setup_environment(args.ffmpeg_dir)

    logging.info("Importing heavy libraries (PyTorch, WhisperX)...")
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    logging.info(f"Starting process for file: {args.audio_file}")
    logging.info(f"Using device: {args.device} | compute_type: {args.compute_type}")

    # 1. Transcribe with original whisper (batched)
    logging.info(f"Loading transcription model from: {args.model_path}")
    model = whisperx.load_model(args.model_path, args.device, compute_type=args.compute_type, local_files_only=True)

    logging.info("Loading audio...")
    audio = whisperx.load_audio(args.audio_file)

    logging.info("Transcribing audio...")
    result = model.transcribe(audio, batch_size=args.batch_size)
    logging.info("Transcription completed.")
    logging.debug(f"Segments before alignment: {result['segments']}")

    # Clean up model to free GPU resources
    if args.low_mem:
        logging.info("Cleaning up transcription model and freeing VRAM (--low-mem active)...")
        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # 2. Align whisper output
    logging.info(f"Loading alignment model: {args.align_model}")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=args.device,
        model_name=args.align_model
    )
    logging.info("Aligning transcription segments...")
    result = whisperx.align(result["segments"], model_a, metadata, audio, args.device, return_char_alignments=False)
    logging.info("Alignment completed.")
    logging.debug(f"Segments after alignment: {result['segments']}")

    # Clean up alignment model to free GPU resources
    if args.low_mem:
        logging.info("Cleaning up alignment model and freeing VRAM (--low-mem active)...")
        del model_a
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # 3. Assign speaker labels (Diarization)
    logging.info(f"Loading diarization pipeline from: {args.diarize_config}")
    diarize_model = DiarizationPipeline(
        model_name=args.diarize_config,
        device=args.device
    )

    logging.info(f"Performing speaker diarization (min_speakers={args.min_speakers}, max_speakers={args.max_speakers})...")
    diarize_segments = diarize_model(audio, min_speakers=args.min_speakers, max_speakers=args.max_speakers)
    logging.info("Speaker diarization completed.")
    logging.debug(f"Diarization output segments:\n{diarize_segments}")

    logging.info("Assigning speakers to words...")
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # Clean up diarization model to free GPU resources
    if args.low_mem:
        logging.info("Cleaning up diarization model and freeing VRAM (--low-mem active)...")
        del diarize_model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    logging.info("Process finished successfully!")

    print("\n--- Final Transcribed Segments with Speaker IDs ---")
    for segment in result["segments"]:
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "")
        print(f"[{start:06.2f}s - {end:06.2f}s] Speaker {speaker}: {text}")

if __name__ == "__main__":
    main()
