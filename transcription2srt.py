import argparse
import re
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convert WhisperX transcription text to SubRip (.srt) subtitle format.")
    parser.add_argument("input_file", help="Path to the input transcription text file.")
    parser.add_argument("--output-file", "-o", help="Optional path to output the .srt file. If omitted, uses the input filename with .srt extension.", default=None)
    return parser.parse_args()

def format_srt_time(seconds_float):
    """Converts floating point seconds into HH:MM:SS,mmm string"""
    hours = int(seconds_float // 3600)
    minutes = int((seconds_float % 3600) // 60)
    seconds = int(seconds_float % 60)
    milliseconds = int(round((seconds_float - int(seconds_float)) * 1000))
    # Cap milliseconds at 999
    if milliseconds >= 1000:
        milliseconds = 999
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def parse_transcription(file_path):
    """
    Parses the transcription text file and returns a list of segments:
    [{'start': float, 'end': float, 'text': str}]
    """
    prefix_pattern = re.compile(r"^\[([\d:.]+)s?\s*(?:-->|-)\s*([\d:.]+)s?\](?:.*?:\s*)?(.*)$")
    segments = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        current_start, current_end = None, None
        current_text = ""
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            match = prefix_pattern.match(stripped)
            if match:
                # Save previous block
                if current_start is not None:
                    segments.append({
                        'start': current_start, 
                        'end': current_end, 
                        'text': current_text.strip()
                    })
                
                start_str, end_str, text_part = match.groups()
                current_start = float(start_str) if ':' not in start_str else sum(x * float(t) for x, t in zip([3600, 60, 1], start_str.split(":")))
                current_end = float(end_str) if ':' not in end_str else sum(x * float(t) for x, t in zip([3600, 60, 1], end_str.split(":")))
                current_text = text_part
            else:
                # Continuation of multi-line text block
                if current_start is not None:
                    current_text += "\n" + stripped # Preserve newlines for SRT!
        
        # Save last block
        if current_start is not None:
            segments.append({
                'start': current_start, 
                'end': current_end, 
                'text': current_text.strip()
            })
            
    return segments

def main():
    args = parse_args()
    input_path = Path(args.input_file)
    
    if not input_path.is_file():
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)
        
    output_path = args.output_file
    if output_path is None:
        output_path = input_path.with_suffix('.srt')
        
    segments = parse_transcription(input_path)
    if not segments:
        print(f"Error: No valid timestamps found in '{input_path}'. Are you sure this is a transcription file?", file=sys.stderr)
        sys.exit(1)
        
    print(f"Parsed {len(segments)} segments. Generating SRT...")
    
    with open(output_path, "w", encoding="utf-8") as out:
        for i, seg in enumerate(segments, 1):
            start_str = format_srt_time(seg['start'])
            end_str = format_srt_time(seg['end'])
            
            out.write(f"{i}\n")
            out.write(f"{start_str} --> {end_str}\n")
            out.write(f"{seg['text']}\n\n")
            
    print(f"Successfully wrote {output_path}")

if __name__ == "__main__":
    main()
