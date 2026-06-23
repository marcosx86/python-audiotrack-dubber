import argparse
import logging
import re
import os
import sys
import time
import statistics
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Condenses translated transcriptions to fit within strict audio duration windows using an LLM."
    )
    parser.add_argument("translated_transcription", help="Path to the translated WhisperX .txt file")
    parser.add_argument("--output-file", "-o", default="condensed_translation.txt", help="Path to save the condensed transcription")
    parser.add_argument("--max-chars-per-sec", type=float, default=15.0, help="Maximum allowed Portuguese characters per second of audio (default: 15.0)")
    parser.add_argument("--endpoint", default="http://localhost:1234/v1", help="OpenAI-compatible API endpoint (default: LM Studio's localhost:1234/v1)")
    parser.add_argument("--api-key", default=None, help="API Key for the endpoint. Evaluated as: 1) value provided, 2) LM_STUDIO_API_KEY environment variable, 3) no Bearer token (no bearer authorization required).")
    parser.add_argument("--model", default="local-model", help="Model name to request from the API (LM Studio usually ignores this but API needs it)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()

def parse_time(time_str):
    if ':' in time_str:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        return float(time_str.replace('s', ''))

def condense_text(client, model, original_text, target_chars):
    system_prompt = (
        "You are an expert dubbing scriptwriter. Your job is to rewrite Portuguese subtitles "
        "to be shorter, punchier, and faster to speak, without losing the core meaning. "
        "You must strictly obey the character limit provided. OUTPUT ONLY THE CONDENSED TEXT. "
        "Do not output conversational filler, introductions, quotes, or explanations."
    )
    user_prompt = f"Condense the following text to be strictly under {int(target_chars)} characters:\n\n{original_text}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=300,
        )
        condensed = response.choices[0].message.content.strip()
        # Clean up accidental quotes if the model added them
        if condensed.startswith('"') and condensed.endswith('"'):
            condensed = condensed[1:-1]
        return condensed
    except Exception as e:
        logging.error(f"Failed to condense text: {e}")
        return original_text

def main():
    args = parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - [%(levelname)s] - %(message)s")

    # Resolve API Key securely
    api_key = args.api_key or os.environ.get("LM_STUDIO_API_KEY")

    logging.info(f"Connecting to LLM Endpoint: {args.endpoint}")
    try:
        # The openai python client mandates a non-empty string for the api_key parameter,
        # even for local unauthenticated endpoints. We pass "dummy" if no key is provided,
        # which LM Studio (and most unauthenticated endpoints) will safely ignore.
        client = OpenAI(base_url=args.endpoint, api_key=api_key if api_key else "dummy")
        
        models_response = client.models.list()
        loaded_models = [m.id for m in models_response.data]
        
        if args.model not in loaded_models:
            logging.info(f"Model '{args.model}' not found in loaded models. Triggering dynamic load via warmup request...")
            # Trigger JIT load into VRAM so it doesn't skew our condensation timing metrics
            client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": "warmup"}],
                max_tokens=1
            )
            logging.info(f"Model '{args.model}' successfully loaded into VRAM.")
        else:
            logging.info(f"Model '{args.model}' is already loaded and ready.")
            
    except Exception as e:
        logging.error(f"Failed to connect to LLM at {args.endpoint}. Is LM Studio running? Error: {e}")
        sys.exit(1)

    if not os.path.exists(args.translated_transcription):
        logging.error(f"Input file not found: {args.translated_transcription}")
        sys.exit(1)

    # Robust pattern matching both '00:00:00.000 --> 00:00:00.000' and '000.57s - 001.51s'
    pattern = re.compile(r'^\[([\d:.]+)s?\s*(?:-->|-)\s*([\d:.]+)s?\](?:.*?:\s*)?(.*)$')
    
    lines_processed = 0
    condensed_count = 0
    total_chars_saved = 0
    llm_call_times = []
    
    global_start_time = time.time()

    with open(args.translated_transcription, 'r', encoding='utf-8') as infile, \
         open(args.output_file, 'w', encoding='utf-8') as outfile:
        
        logging.info("Starting text condensation pass...")
        
        for line in infile:
            match = pattern.match(line.strip())
            if not match:
                # Keep unrecognized lines intact (empty lines, headers)
                outfile.write(line + "\n" if not line.endswith("\n") else line)
                continue
                
            start_str, end_str, text = match.groups()
            duration = parse_time(end_str) - parse_time(start_str)
            logging.debug(f"Processing line: {text} duration: {duration}")
            
            if duration <= 0:
                outfile.write(line + "\n")
                continue
                
            target_char_limit = duration * args.max_chars_per_sec
            
            if len(text) > target_char_limit:
                logging.debug(f"[{start_str} - {end_str}] Too long ({len(text)} > {int(target_char_limit)}). Condensing...")
                
                llm_t0 = time.time()
                condensed = condense_text(client, args.model, text, target_char_limit)
                llm_call_times.append(time.time() - llm_t0)
                
                if len(condensed) < len(text):
                    total_chars_saved += (len(text) - len(condensed))
                    condensed_count += 1
                    logging.debug(f"  Original : {text}")
                    logging.debug(f"  Condensed: {condensed} ({len(condensed)} chars)")
                    
                    # Reconstruct the line
                    outfile.write(f"[{start_str} --> {end_str}] {condensed}\n")
                else:
                    logging.debug("  LLM failed to shorten text. Keeping original.")
                    outfile.write(line + "\n" if not line.endswith("\n") else line)
            else:
                outfile.write(line + "\n" if not line.endswith("\n") else line)
                
            lines_processed += 1

    total_time = time.time() - global_start_time

    logging.info("========================================")
    logging.info("CONDENSATION COMPLETE")
    logging.info("========================================")
    logging.info(f"Lines processed:    {lines_processed}")
    logging.info(f"Lines condensed:    {condensed_count}")
    logging.info(f"Total chars saved:  {total_chars_saved}")
    logging.info(f"Output saved to:    {args.output_file}")
    
    logging.info("----------------------------------------")
    logging.info("PERFORMANCE METRICS")
    logging.info("----------------------------------------")
    logging.info(f"Total Script Time:  {total_time:.2f}s")
    if llm_call_times:
        mean_time = statistics.mean(llm_call_times)
        median_time = statistics.median(llm_call_times)
        min_time = min(llm_call_times)
        max_time = max(llm_call_times)
        logging.info(f"LLM Calls Made:     {len(llm_call_times)}")
        logging.info(f"Mean LLM Time:      {mean_time:.2f}s per call")
        logging.info(f"Median LLM Time:    {median_time:.2f}s per call")
        logging.info(f"Fastest LLM Call:   {min_time:.2f}s")
        logging.info(f"Slowest LLM Call:   {max_time:.2f}s")
    logging.info("========================================")

if __name__ == "__main__":
    main()
