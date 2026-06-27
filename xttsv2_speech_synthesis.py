import logging

logger = logging.getLogger("WhisperX")
logger.propagate = False

import os
import sys
import re
import shutil
import threading
import queue
import time
import argparse
import tempfile
import subprocess
from pathlib import Path
import statistics

def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize translated audio using Coqui XTTS-v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input files
    parser.add_argument("original_audio", help="Path to the original source audio/video file")
    parser.add_argument("original_transcription", help="Path to the original transcription text file")
    parser.add_argument("translated_transcription", help="Path to the translated transcription text file")
    
    # Output configuration
    parser.add_argument("--output-file", "-o", default="translated_output.wav", help="Path to save the final synthesized audio")
    parser.add_argument("--reference-audio-path", default=None, help="Path to a single clean reference audio file (e.g. narrator_reference.wav) to extract global latents from, bypassing dynamic per-segment extraction.")
    parser.add_argument("--ffmpeg-path", default="ffmpeg", help="Custom path to the ffmpeg executable")
    parser.add_argument("--time-stretch", action="store_true", help="Enable FFmpeg time-stretching (condensation) to force generated audio to fit original timestamps")
    parser.add_argument("--start-sentence", type=int, default=None, help="1-indexed starting sentence to generate (inclusive)")
    parser.add_argument("--end-sentence", type=int, default=None, help="1-indexed ending sentence to generate (inclusive)")
    parser.add_argument("--cooldown", type=float, default=1.5, help="Artificial delay (in seconds) between LLM calls to prevent GPU overheating/BSODs (default: 1.5)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()

def extract_reference_audio(audio_path, start, duration, ffmpeg_path, target_sr=24000):
    """
    Extracts a segment of audio using FFmpeg and returns the path to a temporary wav file.
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
    import torchaudio

    generated_dur = waveform.shape[1] / sr
    if generated_dur <= target_dur * 1.02: # Allow tiny 2% margin to avoid unnecessary I/O
        return waveform
        
    tempo = generated_dur / target_dur
    # FFmpeg's atempo accepts values between 0.5 and 100.0.
    tempo = min(tempo, 100.0)
    
    logger.debug(f"[TimeStretch] Compressing {generated_dur:.2f}s to {target_dur:.2f}s (Speed {tempo:.2f}x)")
    
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
    # Matches both format styles, e.g. [041.94s - 055.41s] Speaker SPEAKER_01: Text
    prefix_pattern = re.compile(r"^\[([\d:.]+)s?\s*(?:-->|-)\s*([\d:.]+)s?\](?:.*?:\s*)?(.*)$")
    
    def parse_file(path):
        segments = []
        with open(path, "r", encoding="utf-8") as f:
            current_start, current_end = None, None
            current_text = ""
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                match = prefix_pattern.match(stripped)
                if match:
                    if current_start is not None:
                        segments.append({'start': current_start, 'end': current_end, 'text': current_text.strip()})
                    
                    start_str, end_str, text_part = match.groups()
                    current_start = float(start_str) if ':' not in start_str else sum(x * float(t) for x, t in zip([3600, 60, 1], start_str.split(":")))
                    current_end = float(end_str) if ':' not in end_str else sum(x * float(t) for x, t in zip([3600, 60, 1], end_str.split(":")))
                    current_text = text_part
                else:
                    if current_start is not None:
                        current_text += " " + stripped
            
            if current_start is not None:
                segments.append({'start': current_start, 'end': current_end, 'text': current_text.strip()})
        return segments

    orig_segs = parse_file(orig_path)
    trans_segs = parse_file(trans_path)
    
    results = []
    # Combine them safely, avoiding desyncs from arbitrary newlines
    for orig, trans in zip(orig_segs, trans_segs):
        if not orig['text'] or not trans['text']:
            continue
        results.append({
            'start': orig['start'],
            'end': orig['end'],
            'duration': orig['end'] - orig['start'],
            'orig_text': orig['text'],
            'trans_text': trans['text']
        })
        
    return results

def audio_extractor_worker(task_queue, segments, original_audio, ffmpeg_path, skip_extraction=False):
    """
    Background worker that extracts audio segments via FFmpeg ahead of time.
    Places a dict containing the segment data and the temporary wav path into the queue.
    """
    for idx, seg in enumerate(segments):
        seg_start = seg['start']
        duration = seg['duration']
        
        try:
            if skip_extraction:
                wav_path = None
            else:
                t0 = time.time()
                # Extract at 24kHz for XTTSv2 prompt matching
                wav_path = extract_reference_audio(original_audio, seg_start, duration, ffmpeg_path, target_sr=24000)
                logger.debug(f"[Producer] Extracted segment {idx+1} in {time.time() - t0:.3f}s (Queue size: {task_queue.qsize()})")
            
            # Block if the queue is full (maxsize reached)
            task_queue.put({
                'idx': idx,
                'seg': seg,
                'wav_path': wav_path,
                'error': None
            })
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
    logger.setLevel(log_level)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
        logger.addHandler(ch)

    # Verify input files
    for path in [args.original_audio, args.original_transcription, args.translated_transcription]:
        if not Path(path).is_file():
            logger.error(f"File not found: {path}")
            sys.exit(1)

    logger.info("Parsing transcriptions...")
    segments = parse_transcriptions(args.original_transcription, args.translated_transcription)
    if not segments:
        logger.error("No valid matching segments found in transcriptions.")
        sys.exit(1)

    start_idx = max(0, args.start_sentence - 1) if args.start_sentence is not None else 0
    end_idx = args.end_sentence if args.end_sentence is not None else len(segments)
    is_trimmed = args.start_sentence is not None or args.end_sentence is not None

    # Store global offset to maintain proper logging ID numbers
    global_offset = start_idx
    segments = segments[start_idx:end_idx]
    
    if not segments:
        logger.error(f"No segments left after trimming from {args.start_sentence} to {args.end_sentence}.")
        sys.exit(1)
        
    logger.info(f"Loaded and sliced {len(segments)} segments for synthesis.")

    # Loading heavy modules after our argparse intro
    logger.debug("Loading torch module...")
    import torch
    logger.debug("Loading torchaudio module...")
    import torchaudio
    logger.debug("Loading NVIDIA NeMo's text processing module...")
    from nemo_text_processing.text_normalization.normalize import Normalizer
    logger.debug("Loading Coqui TTS module...")
    from TTS.api import TTS
    logger.debug("Loading XTTS config modules...")
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig

    # Tell PyTorch to trust the XTTS config class (fixes deserialization security errors)
    logger.debug("Enabling XTTS config trust on torch library...")
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

    # Fix Windows DLL loading for torchcodec (FFmpeg shared binaries) ONLY AFTER PyTorch is loaded
    if sys.platform == "win32":
        logger.debug("Checking and fixing Windows DLL loading for torchcodec...")
        ffmpeg_exe = shutil.which(args.ffmpeg_path)
        if ffmpeg_exe:
            logger.debug(f"FFmpeg found at: {ffmpeg_exe}")
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            try:
                os.add_dll_directory(ffmpeg_dir)
                logger.debug(f"Successfully registered FFmpeg DLL directory: {ffmpeg_dir}")
            except Exception as e:
                logger.warning(f"Failed to register DLL directory {ffmpeg_dir}: {e}")

    # TODO: Parametrize this model string later via argparser
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    logger.info(f"Initializing Coqui XTTS-v2 AutoModel with: {model_name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        tts = TTS(model_name=model_name, progress_bar=False).to(device)
        xtts = tts.synthesizer.tts_model
    except Exception as e:
        logger.error(f"Failed to initialize XTTS-v2: {e}")
        sys.exit(1)

    global_gpt_cond_latent = None
    global_speaker_embedding = None
    if args.reference_audio_path:
        if not Path(args.reference_audio_path).is_file():
            logger.error(f"Reference audio not found: {args.reference_audio_path}")
            sys.exit(1)
        logger.info(f"Computing global voice latents from: {args.reference_audio_path}")
        try:
            global_gpt_cond_latent, global_speaker_embedding = xtts.get_conditioning_latents(
                audio_path=[args.reference_audio_path],
                gpt_cond_len=30,
            )
        except Exception as e:
            logger.error(f"Failed to compute global latents: {e}")
            sys.exit(1)

    logger.info("Initializing NeMo Text Normalizer for pt_BR...")
    try:
        nemo_normalizer = Normalizer(input_case='cased', lang='pt')
    except Exception as e:
        logger.error(f"Failed to initialize NeMo Normalizer: {e}")
        sys.exit(1)

    target_sr = 24000  # Default XTTSv2 native sample rate

    # Initialize task queue and start background extraction thread
    task_queue = queue.Queue(maxsize=5) # Pre-fetch up to 5 segments to save RAM/Disk
    
    extractor_thread = threading.Thread(
        target=audio_extractor_worker,
        args=(task_queue, segments, args.original_audio, args.ffmpeg_path, bool(args.reference_audio_path)),
        daemon=True
    )
    logger.info("Starting background audio extraction thread...")
    extractor_thread.start()

    final_audio_pieces = []
    # Initialize current_time to the start of the first sliced segment to prevent massive silence padding
    current_time = segments[0]['start'] if segments else 0.0

    logger.info("Starting XTTS-v2 zero-shot TTS generation...")
    global_start_time = time.time()
    segment_processing_times = []

    while True:
        wait_t0 = time.time()
        task = task_queue.get()
        if task is None:
            break # All segments processed
            
        segment_t0 = time.time()
        wait_time = segment_t0 - wait_t0
        idx = task['idx']
        seg = task['seg']
        reference_audio_path = task['wav_path']
        extract_error = task['error']
        
        start = seg['start']
        end = seg['end']
        duration = seg['duration']
        orig_text = seg['orig_text']
        trans_text = seg['trans_text']
        
        # Normalize text to Portuguese before processing
        logger.debug(f"Calling NeMo to normalize target text: {trans_text}")
        try:
            trans_text = nemo_normalizer.normalize(trans_text, verbose=False)
        except Exception as e:
            logger.warning(f"NeMo normalization failed for '{trans_text}'. Using original text. Error: {e}")
        
        logger.info(f"[{global_offset + idx + 1}/{global_offset + len(segments)}{' (Trimmed)' if is_trimmed else ''}] Synthesizing clip: [{start:.2f}s - {end:.2f}s]")
        logger.debug(f"[Consumer] Waited {wait_time:.3f}s for Producer queue")
        logger.debug(f"  Reference Text: {orig_text}")
        logger.debug(f"  Normalized Text:    {trans_text}")

        if extract_error and not args.reference_audio_path:
            logger.error(f"  Failed to extract reference audio: {extract_error}")
            logger.warning("  Skipping this segment to continue pipeline.")
            continue

        # 1. Padding with Silence to emulate absolute timestamps
        if start > current_time:
            silence_dur = start - current_time
            logger.debug(f"  Padding with {silence_dur:.2f}s of silence to align timeline.")
            silence_samples = int(silence_dur * target_sr)
            final_audio_pieces.append(torch.zeros(1, silence_samples))
            current_time = start

        # 2. Generate Audio
        try:
            if global_gpt_cond_latent is not None:
                gpt_cond_latent = global_gpt_cond_latent
                speaker_embedding = global_speaker_embedding
            else:
                logger.debug(f"[Consumer] Computing voice latents from reference audio...")
                # Compute latents from dynamically extracted prompt audio
                gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(
                    audio_path=[reference_audio_path],
                    gpt_cond_len=30, # Max conditioning length is 30s
                )
            
            infer_t0 = time.time()
            
            logger.debug(f"[Consumer] Calling xtts.inference...")
            out = xtts.inference(
                text=trans_text,
                language="pt",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                speed=1.0
            )

            infer_time = time.time() - infer_t0
            logger.debug(f"[Consumer] Finished inference in {infer_time:.3f}s")

            if args.cooldown > 0:
                logging.debug(f"[Consumer] Cooling down for {args.cooldown}s to protect hardware...")
                time.sleep(args.cooldown)

            cpu_t0 = time.time()
            # Convert NumPy array to PyTorch tensor and add channel dimension
            tts_speech = torch.from_numpy(out["wav"]).unsqueeze(0)
            logger.debug(f"[Consumer] Converted numpy output to CPU tensor in {time.time() - cpu_t0:.3f}s")
            
            # --- START TIMELINE MODEL ---
            generated_dur = tts_speech.shape[1] / target_sr
            target_dur = end - start
            
            if generated_dur > target_dur:
                if args.time_stretch:
                    # 3a. Stretch audio if it exceeds strict window
                    stretched_speech = apply_time_stretch_ffmpeg(tts_speech, target_sr, target_dur, args.ffmpeg_path)
                    final_audio_pieces.append(stretched_speech)
                    logger.debug(f"  Time-stretched audio from {generated_dur:.2f}s down to {target_dur:.2f}s.")
                    
                    generated_dur = stretched_speech.shape[1] / target_sr
                    current_time = end
                    logger.debug(f"  Timeline strictly advanced to {current_time:.2f}s.")
                else:
                    # 3b. Accept timeline drift (Condensation disabled)
                    final_audio_pieces.append(tts_speech)
                    current_time += generated_dur
                    logger.warning(f"  Timeline drifted to {current_time:.2f}s (overshot by {generated_dur - target_dur:.2f}s).")
            else:
                # 3c. Pad with silence if it's shorter than strict window
                final_audio_pieces.append(tts_speech)
                shortfall = target_dur - generated_dur
                silence_samples = int(shortfall * target_sr)
                if silence_samples > 0:
                    final_audio_pieces.append(torch.zeros(1, silence_samples))
                logger.debug(f"  Padded tail with {shortfall:.2f}s of silence.")
                current_time = end
                logger.debug(f"  Timeline strictly advanced to {current_time:.2f}s.")
            # --- END TIMELINE MODEL ---

            segment_processing_time = time.time() - segment_t0
            segment_processing_times.append(segment_processing_time)
            logger.info(f"  Segment {idx+1} processed in {segment_processing_time:.2f}s")
        except Exception as e:
            logger.error(f"  TTS generation failed for segment {idx+1}: {e}")
            logger.warning("  Skipping this segment to continue pipeline.")
            continue
        finally:
            if reference_audio_path and os.path.exists(reference_audio_path):
                os.remove(reference_audio_path)

    # Assemble and save
    logger.info("Concatenating all segments and silence buffers...")
    if final_audio_pieces:
        final_waveform = torch.cat(final_audio_pieces, dim=1)
        torchaudio.save(args.output_file, final_waveform, target_sr)
        logger.info("==================================================")
        logger.info(f"Synthesized Output Saved: {args.output_file}")
        logger.info(f"Final Track Duration: {final_waveform.shape[1] / target_sr:.2f} seconds")
        logger.info("==================================================")
    else:
        logger.error("No audio was generated!")
        
    total_time = time.time() - global_start_time
    
    if segment_processing_times:
        mean_time = statistics.mean(segment_processing_times)
        median_time = statistics.median(segment_processing_times)
        min_time = min(segment_processing_times)
        max_time = max(segment_processing_times)
        
        logger.info("=" * 50)
        logger.info("SYNTHESIS PERFORMANCE STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total Pipeline Time:       {total_time:.2f}s")
        logger.info(f"Total Segments Processed:  {len(segment_processing_times)}")
        logger.info(f"Mean Time per Segment:     {mean_time:.2f}s")
        logger.info(f"Median Time per Segment:   {median_time:.2f}s")
        logger.info(f"Fastest Segment:           {min_time:.2f}s")
        logger.info(f"Slowest Segment:           {max_time:.2f}s")
        logger.info("=" * 50)

if __name__ == "__main__":
    main()
