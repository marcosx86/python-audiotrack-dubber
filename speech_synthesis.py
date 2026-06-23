import logging
import os
import sys
import re
import shutil
import logging
import threading
import queue
import time
import argparse
import tempfile
import subprocess
from pathlib import Path

import torch
import torchaudio

def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize translated audio using CosyVoice with GGUF support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input files
    parser.add_argument("original_audio", help="Path to the original source audio/video file")
    parser.add_argument("original_transcription", help="Path to the original transcription text file")
    parser.add_argument("translated_transcription", help="Path to the translated transcription text file")
    
    # Model configuration
    parser.add_argument("--model-dir", required=True, help="Path to the CosyVoice model directory (e.g. D:\\CosyVoice\\pretrained_models\\CosyVoice2-0.5B)")
    
    # Output configuration
    parser.add_argument("--output-file", "-o", default="translated_output.wav", help="Path to save the final synthesized audio")
    parser.add_argument("--ffmpeg-path", default="ffmpeg", help="Custom path to the ffmpeg executable")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision for CosyVoice inference to speed up generation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()

def extract_reference_audio(audio_path, start, duration, ffmpeg_path, target_sr=16000):
    """
    Extracts a segment of audio using FFmpeg and returns the path to a 16kHz temporary wav file.
    The caller is responsible for deleting the file after use.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
        
    cmd = [
        ffmpeg_path, "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", str(audio_path),
        "-vn",
        "-ac", "1",
        "-ar", str(target_sr),
        tmp_name
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
    return tmp_name

def apply_time_stretch_ffmpeg(waveform, sr, target_dur, ffmpeg_path):
    """
    If the waveform is longer than target_dur, stretches it using FFmpeg's atempo filter.
    Returns the stretched tensor.
    """
    generated_dur = waveform.shape[1] / sr
    if generated_dur <= target_dur * 1.02: # Allow tiny 2% margin to avoid unnecessary I/O
        return waveform
        
    tempo = generated_dur / target_dur
    # FFmpeg's atempo accepts values between 0.5 and 100.0.
    tempo = min(tempo, 100.0)
    
    logging.debug(f"[TimeStretch] Compressing {generated_dur:.2f}s to {target_dur:.2f}s (Speed {tempo:.2f}x)")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        in_name = tmp_in.name
        out_name = tmp_out.name
        
    try:
        torchaudio.save(in_name, waveform, sr)
        
        cmd = [
            ffmpeg_path, "-y",
            "-i", in_name,
            "-filter:a", f"atempo={tempo}",
            out_name
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        stretched_wav, _ = torchaudio.load(out_name)
        return stretched_wav
    finally:
        if os.path.exists(in_name):
            os.remove(in_name)
        if os.path.exists(out_name):
            os.remove(out_name)

def parse_transcriptions(orig_path, trans_path):
    """
    Parses both original and translated transcriptions.
    Returns a list of dicts: {'start': float, 'end': float, 'duration': float, 'orig_text': str, 'trans_text': str}
    """
    # Matches: [041.94s - 055.41s] Speaker SPEAKER_01: The actual text
    prefix_pattern = re.compile(r"^\[(\d+\.\d+)s\s*-\s*(\d+\.\d+)s\](?:.*?:\s*)(.*)$")
    
    segments = []
    
    with open(orig_path, "r", encoding="utf-8") as fo, open(trans_path, "r", encoding="utf-8") as ft:
        for orig_line, trans_line in zip(fo, ft):
            orig_line = orig_line.strip()
            trans_line = trans_line.strip()
            
            if not orig_line or not trans_line:
                continue
                
            orig_match = prefix_pattern.match(orig_line)
            trans_match = prefix_pattern.match(trans_line)
            
            if orig_match and trans_match:
                start = float(orig_match.group(1))
                end = float(orig_match.group(2))
                orig_text = orig_match.group(3)
                trans_text = trans_match.group(3)
                
                # Only add if there's actual text
                if orig_text.strip() and trans_text.strip():
                    segments.append({
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "orig_text": orig_text,
                        "trans_text": trans_text
                    })
    return segments

def audio_extractor_worker(task_queue, segments, original_audio, ffmpeg_path):
    """
    Background worker that extracts audio segments via FFmpeg ahead of time.
    Places a dict containing the segment data and the temporary wav path into the queue.
    """
    for idx, seg in enumerate(segments):
        seg_start = seg['start']
        duration = seg['duration']
        
        try:
            t0 = time.time()
            # Extract at 16kHz for CosyVoice prompts
            wav_path = extract_reference_audio(original_audio, seg_start, duration, ffmpeg_path, target_sr=16000)
            
            # Block if the queue is full (maxsize reached)
            task_queue.put({
                'idx': idx,
                'seg': seg,
                'wav_path': wav_path,
                'error': None
            })
            logging.debug(f"[Producer] Extracted segment {idx+1} in {time.time() - t0:.3f}s (Queue size: {task_queue.qsize()})")
        except Exception as e:
            # Pass the error down the queue so the main thread can handle/log it safely
            task_queue.put({
                'idx': idx,
                'seg': seg,
                'wav_path': None,
                'error': e
            })
            
    # Send a poison pill to indicate completion
    task_queue.put(None)

def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - [%(levelname)s] - %(message)s")

    # Fix Windows DLL loading for torchcodec (FFmpeg shared binaries)
    if sys.platform == "win32":
        ffmpeg_exe = shutil.which(args.ffmpeg_path)
        if ffmpeg_exe:
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            try:
                os.add_dll_directory(ffmpeg_dir)
                logging.debug(f"Automatically registered FFmpeg DLL directory: {ffmpeg_dir}")
            except Exception as e:
                logging.warning(f"Failed to register DLL directory {ffmpeg_dir}: {e}")

    # Verify input files
    for path in [args.original_audio, args.original_transcription, args.translated_transcription]:
        if not Path(path).is_file():
            logging.error(f"File not found: {path}")
            sys.exit(1)
            
    if not Path(args.model_dir).is_dir():
        logging.error(f"Model directory not found: {args.model_dir}")
        sys.exit(1)

    logging.info("Parsing transcriptions...")
    segments = parse_transcriptions(args.original_transcription, args.translated_transcription)
    logging.info(f"Found {len(segments)} valid synchronized segments.")

    if not segments:
        logging.warning("No segments found to process.")
        sys.exit(0)

    # Lazy-load CosyVoice to avoid failing fast on arg checks
    logging.debug("Loading CosyVoice modules...")
    try:
        # Add model dir and its third_party dependencies to path
        # Assuming the script runs from WhisperX but model_dir might be D:\CosyVoice\pretrained_models\...
        # We need the base CosyVoice repo path to import correctly.
        base_cosyvoice_dir = str(Path(args.model_dir).parent.parent) # D:\CosyVoice
        sys.path.append(base_cosyvoice_dir)
        sys.path.append(os.path.join(base_cosyvoice_dir, 'third_party', 'Matcha-TTS'))
        
        from cosyvoice.cli.cosyvoice import AutoModel
    except ImportError as e:
        logging.error(f"Failed to import CosyVoice. Ensure you have the library installed. Error: {e}")
        sys.exit(1)
        
    logging.info(f"Initializing CosyVoice AutoModel with model: {args.model_dir} (fp16={args.fp16})")
    
    try:
        cosyvoice = AutoModel(model_dir=args.model_dir, fp16=args.fp16)
    except Exception as e:
        logging.error(f"Failed to initialize CosyVoice: {e}")
        sys.exit(1)

    target_sr = 22050  # Typical CosyVoice sample rate, updated below if model has it
    if hasattr(cosyvoice, 'sample_rate'):
        target_sr = cosyvoice.sample_rate

    # Initialize task queue and start background extraction thread
    task_queue = queue.Queue(maxsize=5) # Pre-fetch up to 5 segments to save RAM/Disk
    
    extractor_thread = threading.Thread(
        target=audio_extractor_worker,
        args=(task_queue, segments, args.original_audio, args.ffmpeg_path),
        daemon=True
    )
    logging.info("Starting background audio extraction thread...")
    extractor_thread.start()

    final_audio_pieces = []
    current_time = 0.0

    logging.info("Starting zero-shot TTS generation...")

    while True:
        wait_t0 = time.time()
        task = task_queue.get()
        if task is None:
            break # All segments processed
            
        wait_time = time.time() - wait_t0
        idx = task['idx']
        seg = task['seg']
        reference_audio_path = task['wav_path']
        extract_error = task['error']
        
        start = seg['start']
        end = seg['end']
        duration = seg['duration']
        orig_text = seg['orig_text']
        trans_text = seg['trans_text']
        
        logging.info(f"[{idx+1}/{len(segments)}] Synthesizing clip: [{start:.2f}s - {end:.2f}s]")
        logging.debug(f"[Consumer] Waited {wait_time:.3f}s for Producer queue")
        logging.debug(f"  Reference Text: {orig_text}")
        logging.debug(f"  Target Text:    {trans_text}")

        if extract_error:
            logging.error(f"  Failed to extract reference audio: {extract_error}")
            logging.warning("  Skipping this segment to continue pipeline.")
            continue

        # 1. Padding with Silence to emulate absolute timestamps
        if start > current_time:
            silence_dur = start - current_time
            logging.debug(f"  Padding with {silence_dur:.2f}s of silence to align timeline.")
            silence_samples = int(silence_dur * target_sr)
            final_audio_pieces.append(torch.zeros(1, silence_samples))
            current_time = start

        # 2. Generate Audio
        try:
            # cosyvoice.inference_zero_shot returns either a dictionary or a generator
            logging.debug(f"[Consumer] Calling cosyvoice.inference_zero_shot...")
            infer_t0 = time.time()
            output = cosyvoice.inference_zero_shot(
                tts_text=trans_text,
                prompt_text=orig_text,
                prompt_wav=reference_audio_path,
                text_frontend=False
            )
            
            # If it yields chunks (streaming), concatenate them
            if hasattr(output, '__iter__') and not isinstance(output, dict):
                logging.debug(f"[Consumer] Output is generator, awaiting GPU chunks...")
                tts_speech = torch.cat([chunk['tts_speech'] for chunk in output], dim=1)
            else:
                logging.debug(f"[Consumer] Output has single chunk, assigning...")
                tts_speech = output['tts_speech']

            infer_time = time.time() - infer_t0
            logging.debug(f"[Consumer] Finished inference and concatenation in {infer_time:.3f}s")
            
            cpu_t0 = time.time()
            final_audio_pieces.append(tts_speech.cpu())
            logging.debug(f"[Consumer] Copied processed tensor to CPU in {time.time() - cpu_t0:.3f}s")
            
            # --- START STRICT TIMELINE MODEL ---
            generated_dur = tts_speech.shape[1] / target_sr
            target_dur = end - start
            
            if generated_dur > target_dur:
                # 3a. Stretch audio if it exceeds strict window
                stretched_speech = apply_time_stretch_ffmpeg(tts_speech.cpu(), target_sr, target_dur, args.ffmpeg_path)
                final_audio_pieces[-1] = stretched_speech
                logging.debug(f"  Time-stretched audio from {generated_dur:.2f}s down to {target_dur:.2f}s.")
                
                # Update generated duration after stretch
                generated_dur = stretched_speech.shape[1] / target_sr
            
            if generated_dur < target_dur:
                # 3b. Pad with silence if it's shorter than strict window
                shortfall = target_dur - generated_dur
                silence_samples = int(shortfall * target_sr)
                if silence_samples > 0:
                    final_audio_pieces.append(torch.zeros(1, silence_samples))
                logging.debug(f"  Padded tail with {shortfall:.2f}s of silence.")
                
            # 4. Advance timeline strictly to the end of the original segment
            current_time = end
            logging.debug(f"  Timeline strictly advanced to {current_time:.2f}s.")
            # --- END STRICT TIMELINE MODEL ---

        except Exception as e:
            logging.error(f"  TTS generation failed for segment {idx+1}: {e}")
            logging.warning("  Skipping this segment to continue pipeline.")
            continue
        finally:
            if reference_audio_path and os.path.exists(reference_audio_path):
                os.remove(reference_audio_path)

    # Assemble and save
    logging.info("Concatenating all segments and silence buffers...")
    if final_audio_pieces:
        final_waveform = torch.cat(final_audio_pieces, dim=1)
        torchaudio.save(args.output_file, final_waveform, target_sr)
        logging.info("==================================================")
        logging.info(f"Synthesized Output Saved: {args.output_file}")
        logging.info(f"Final Track Duration: {final_waveform.shape[1] / target_sr:.2f} seconds")
        logging.info("==================================================")
    else:
        logging.error("No audio was generated!")

if __name__ == "__main__":
    main()
