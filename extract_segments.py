import os
import sys
import re
import subprocess
import logging
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and concatenate audio segments for a target speaker from a transcription file for TTS training/reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("transcript_file", help="Path to the transcription text file containing speaker timestamps")
    parser.add_argument("audio_file", help="Path to the original audio/video file")
    parser.add_argument("--target-speaker", "-s", default="SPEAKER_01", help="The speaker ID to extract segments for")
    parser.add_argument("--min-duration", "-d", type=float, default=3.0, help="Minimum segment duration in seconds to include")
    parser.add_argument("--output-dir", "-o", default="voice_extract", help="Directory to save the segments and final clip")
    parser.add_argument("--output-name", "-n", default="narrator_reference.wav", help="Filename of the final concatenated WAV file")
    parser.add_argument("--sample-rate", "-r", type=int, default=24000, help="Output sample rate (Hz) for TTS processing")
    parser.add_argument("--channels", "-c", type=int, default=1, help="Output audio channels (1=mono, 2=stereo)")
    parser.add_argument("--ffmpeg-path", default="ffmpeg", help="Custom path to the ffmpeg executable")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()

def setup_environment(ffmpeg_path):
    # Check if ffmpeg is accessible
    try:
        subprocess.run([ffmpeg_path, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        logging.error(
            f"FFmpeg executable '{ffmpeg_path}' not found.\n"
            f"Please make sure FFmpeg is installed and added to your system PATH, or specify its location using --ffmpeg-path."
        )
        sys.exit(1)

def extract_voice():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - [%(levelname)s] - %(message)s")

    transcript_path = Path(args.transcript_file)
    audio_path = Path(args.audio_file)

    # Verify input files
    if not transcript_path.is_file():
        logging.error(f"Transcription file not found: {args.transcript_file}")
        sys.exit(1)
    if not audio_path.is_file():
        logging.error(f"Audio/Video file not found: {args.audio_file}")
        sys.exit(1)

    setup_environment(args.ffmpeg_path)

    output_dir = Path(args.output_dir)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse transcription file and filter segments
    logging.info(f"Parsing transcription file: {transcript_path}")
    pattern = re.compile(r"\[(\d+\.\d+)s\s*-\s*(\d+\.\d+)s\]\s*Speaker\s*(\S+):")
    
    segments = []
    total_parsed_duration = 0.0

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            m = pattern.search(line)
            if not m:
                continue

            start = float(m.group(1))
            end = float(m.group(2))
            speaker = m.group(3)
            duration = end - start

            if speaker == args.target_speaker:
                if duration >= args.min_duration:
                    segments.append((start, duration, line.strip()))
                    total_parsed_duration += duration
                else:
                    logging.debug(f"Line {line_num}: Skipped segment for {speaker} (duration {duration:.2f}s < {args.min_duration}s)")

    if not segments:
        logging.warning(f"No segments found matching speaker '{args.target_speaker}' with duration >= {args.min_duration}s.")
        sys.exit(0)

    logging.info(f"Selected {len(segments)} segments for speaker '{args.target_speaker}' (Total duration: {total_parsed_duration:.2f}s)")

    # 2. Extract segment clips using FFmpeg
    logging.info(f"Extracting clips to {segments_dir}...")
    concat_file = output_dir / "concat.txt"
    ref_file = output_dir / "narrator_reference.txt"
    
    with open(concat_file, "w", encoding="utf-8") as concat, open(ref_file, "w", encoding="utf-8") as ref:
        current_time_in_output = 0.0
        for idx, (start, duration, text) in enumerate(segments):
            clip_path = segments_dir / f"clip_{idx:05d}.wav"
            
            pct = (idx + 1) / len(segments) * 100
            logging.info(f"[{idx+1}/{len(segments)} - {pct:.1f}%] Extracting {clip_path.name} (start: {start:.2f}s, duration: {duration:.2f}s)")
            logging.debug(f"Transcription snippet: {text}")

            # Extract the spoken text from the original line
            text_match = re.search(r":\s*(.*)$", text)
            spoken_text = text_match.group(1) if text_match else text
            
            # Write updated timestamp and text to reference file
            ref.write(f"[{current_time_in_output:.2f}s - {current_time_in_output + duration:.2f}s] {spoken_text}\n")
            current_time_in_output += duration

            # Fast input-seeking command structure
            cmd = [
                args.ffmpeg_path,
                "-y",
                "-ss", str(start),
                "-t", str(duration),
                "-i", str(audio_path),
                "-vn",
                "-ac", str(args.channels),
                "-ar", str(args.sample_rate),
                str(clip_path)
            ]

            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                concat.write(f"file '{clip_path.resolve()}'\n")
            except subprocess.CalledProcessError as e:
                logging.error(f"FFmpeg failed to extract segment starting at {start}s: {e}")
                sys.exit(1)

    # 3. Concatenate all extracted clips
    output_reference = output_dir / args.output_name
    logging.info(f"Concatenating segments into final reference WAV: {output_reference}")
    
    cmd_concat = [
        args.ffmpeg_path,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_reference)
    ]

    try:
        subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logging.info("Concatenation completed successfully!")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed to concatenate clips: {e}")
        sys.exit(1)

    # Print summary
    print("\n==================================================")
    print("Voice Extraction Summary:")
    print(f"  Target Speaker    : {args.target_speaker}")
    print(f"  Total Segments    : {len(segments)}")
    print(f"  Total Duration    : {total_parsed_duration:.2f} seconds")
    print(f"  Sample Rate       : {args.sample_rate} Hz")
    print(f"  Channels          : {args.channels} ({'mono' if args.channels == 1 else 'stereo'})")
    print(f"  Final Output File : {output_reference.resolve()}")
    print("==================================================")

if __name__ == "__main__":
    extract_voice()