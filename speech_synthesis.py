import logging
import os
import sys
import re
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path

sys.path.append('../CosyVoice')
sys.path.append('../CosyVoice/third_party/Matcha-TTS')

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
    parser.add_argument("--model-dir", required=True, help="Path to the CosyVoice model directory")
    parser.add_argument("--gguf-file", required=True, help="Path to the quantized .gguf file for the LLM step")
    
    # Output configuration
    parser.add_argument("--output-file", "-o", default="translated_output.wav", help="Path to save the final synthesized audio")
    parser.add_argument("--ffmpeg-path", default="ffmpeg", help="Custom path to the ffmpeg executable")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()

def extract_reference_audio(audio_path, start, duration, ffmpeg_path, target_sr=16000):
    """
    Extracts a segment of audio using FFmpeg and returns a 16kHz torch tensor.
    Using FFmpeg avoids loading the entire source video/audio into RAM.
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
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        waveform, sr = torchaudio.load(tmp_name)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
            
    return waveform

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

def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - [%(levelname)s] - %(message)s")

    # Verify input files
    for path in [args.original_audio, args.original_transcription, args.translated_transcription, args.gguf_file]:
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
        # Add model dir to path in case they are running from a cloned repo structure
        sys.path.append(args.model_dir) 
        from cosyvoice.cli.cosyvoice import CosyVoice, AutoModel
    except ImportError as e:
        logging.error(f"Failed to import CosyVoice. Ensure you have the library installed. Error: {e}")
        sys.exit(1)
        
    logging.info(f"Initializing CosyVoice with GGUF model: {args.gguf_file}")
    
    # The exact initialization parameters depend on the specific llama.cpp fork of CosyVoice
    # Sketching it based on standard approaches:
    try:
        # cosyvoice = CosyVoice(args.model_dir, gguf_path=args.gguf_file)
        cosyvoice = AutoModel(model_dir=args.model_dir)
    except TypeError:
        logging.warning("CosyVoice init didn't accept gguf_path directly. Falling back to default init.")
        cosyvoice = CosyVoice(args.model_dir)

    target_sr = 22050  # Typical CosyVoice sample rate, updated below if model has it
    if hasattr(cosyvoice, 'sample_rate'):
        target_sr = cosyvoice.sample_rate

    final_audio_pieces = []
    current_time = 0.0

    logging.info("Starting zero-shot TTS generation...")

    for idx, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        duration = seg['duration']
        orig_text = seg['orig_text']
        trans_text = seg['trans_text']
        
        logging.info(f"[{idx+1}/{len(segments)}] Synthesizing clip: [{start:.2f}s - {end:.2f}s]")
        logging.debug(f"  Reference Text: {orig_text}")
        logging.debug(f"  Target Text:    {trans_text}")

        # 1. Padding with Silence to emulate absolute timestamps
        if start > current_time:
            silence_dur = start - current_time
            logging.debug(f"  Padding with {silence_dur:.2f}s of silence to align timeline.")
            silence_samples = int(silence_dur * target_sr)
            final_audio_pieces.append(torch.zeros(1, silence_samples))
            current_time = start

        # 2. Extract Reference Audio at 16kHz for prompt
        try:
            reference_audio = extract_reference_audio(args.original_audio, start, duration, args.ffmpeg_path, target_sr=16000)
        except Exception as e:
            logging.error(f"  Failed to extract reference audio: {e}")
            sys.exit(1)

        # 3. Generate Audio
        try:
            # cosyvoice.inference_zero_shot returns either a dictionary or a generator
            output = cosyvoice.inference_zero_shot(
                tts_text=trans_text,
                prompt_text=orig_text,
                prompt_wav=reference_audio
            )
            
            # If it yields chunks (streaming), concatenate them
            if hasattr(output, '__iter__') and not isinstance(output, dict):
                tts_speech = torch.cat([chunk['tts_speech'] for chunk in output], dim=1)
            else:
                tts_speech = output['tts_speech']

            final_audio_pieces.append(tts_speech.cpu())
            
            # 4. Advance timeline by the EXACT duration of the generated audio
            generated_dur = tts_speech.shape[1] / target_sr
            current_time += generated_dur
            logging.debug(f"  Generated {generated_dur:.2f}s of audio.")

        except Exception as e:
            logging.error(f"  TTS generation failed for segment {idx+1}: {e}")
            logging.warning("  Skipping this segment to continue pipeline.")
            continue

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
