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
    parser.add_argument("--context-length", type=int, default=2048, help="Context window size to request during JIT load. Drastically reduces VRAM. (default: 2048)")
    parser.add_argument("--cooldown", type=float, default=1.5, help="Artificial delay (in seconds) between LLM calls to prevent GPU overheating/BSODs (default: 1.5)")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM Temperature (creativity). Lower is more strict, higher is more creative. (default: 0.1)")
    parser.add_argument("--maintain-context", action="store_true", help="Tell the LLM via system prompt to creatively paraphrase while maintaining original context instead of strictly cutting.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()

def parse_time(time_str):
    if ':' in time_str:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        return float(time_str.replace('s', ''))

def condense_text(client, model, original_text, target_chars, temperature=0.1, maintain_context=False):
    if maintain_context:
        system_prompt = (
            "You are an expert dubbing scriptwriter. Your job is to rewrite and paraphrase Portuguese subtitles "
            "to fit the allotted time constraint. You must strictly obey the character limit provided. "
            "However, it is critical that you MAINTAIN THE ORIGINAL CONTEXT AND TONE of the phrase. "
            "You may creatively restructure the sentence to achieve this. "
            "OUTPUT ONLY THE REWRITTEN TEXT. Do not output conversational filler, introductions, quotes, or explanations. "
            "CRITICAL RULE: You must output STRICTLY in Portuguese. Do not output any Chinese characters, pinyin, or translation notes."
        )
    else:
        system_prompt = (
            "You are an expert dubbing scriptwriter. Your job is to rewrite Portuguese subtitles "
            "to be shorter, punchier, and faster to speak, without losing the core meaning. "
            "You must strictly obey the character limit provided. OUTPUT ONLY THE CONDENSED TEXT. "
            "Do not output conversational filler, introductions, quotes, or explanations. "
            "CRITICAL RULE: You must output STRICTLY in Portuguese. Do not output any Chinese characters, pinyin, or translation notes."
        )
        
    user_prompt = f"Rewrite the following text to be strictly under {int(target_chars)} characters:\n\n{original_text}"
    
    try:
        logging.debug(f"Sending request to LLM...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=300,
        )
        condensed = response.choices[0].message.content.strip()
        # Clean up accidental quotes if the model added them
        if condensed.startswith('"') and condensed.endswith('"'):
            condensed = condensed[1:-1]
        logging.debug(f"Condensed text from LLM: {condensed}")
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
                max_tokens=1,
                extra_body={
                    "context_length": args.context_length,
                    "n_ctx": args.context_length
                }
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
    # Group 1: The entire prefix (e.g. "[000.57s - 001.51s] Speaker SPEAKER_01: ")
    # Group 2: Start time string
    # Group 3: End time string
    # Group 4: The text to be translated
    pattern = re.compile(r'^(\[([\d:.]+)s?\s*(?:-->|-)\s*([\d:.]+)s?\](?:.*?:\s*)?)(.*)$')
    
    lines_processed = 0
    condensed_count = 0
    total_chars_saved = 0
    llm_call_times = []
    
    global_start_time = time.time()

    with open(args.translated_transcription, 'r', encoding='utf-8') as infile, \
         open(args.output_file, 'w', encoding='utf-8') as outfile:
        
        logging.info("Starting text condensation pass...")
        
        buffered_prefix = None
        buffered_start_str = None
        buffered_end_str = None
        buffered_text = ""
        buffered_duration = 0
        
        def flush_buffer():
            nonlocal lines_processed, condensed_count, total_chars_saved
            if buffered_prefix is None:
                return
                
            text = buffered_text.strip()
            target_char_limit = buffered_duration * args.max_chars_per_sec
            
            if buffered_duration > 0 and len(text) > target_char_limit:
                logging.debug(f"[{buffered_start_str} - {buffered_end_str}] Too long ({len(text)} > {int(target_char_limit)}). Condensing...")
                
                llm_t0 = time.time()
                condensed = condense_text(client, args.model, text, target_char_limit, args.temperature, args.maintain_context)
                llm_call_times.append(time.time() - llm_t0)
                
                if args.cooldown > 0:
                    logging.debug(f"  Cooling down for {args.cooldown}s to protect hardware...")
                    time.sleep(args.cooldown)
                
                if len(condensed) < len(text):
                    total_chars_saved += (len(text) - len(condensed))
                    condensed_count += 1
                    logging.info(f"  Original : {text}")
                    logging.info(f"  Condensed: {condensed} ({len(condensed)} chars)")
                    
                    # Convert any inner newlines from the model to spaces for a clean single-line output
                    condensed_single_line = condensed.replace('\n', ' ')
                    outfile.write(f"{buffered_prefix}{condensed_single_line}\n")
                else:
                    logging.warning("  LLM failed to shorten text (new size is bigger than original sentence). Keeping original.")
                    outfile.write(f"{buffered_prefix}{text}\n")
            else:
                outfile.write(f"{buffered_prefix}{text}\n")
                
            lines_processed += 1

        for line in infile:
            stripped = line.strip()
            if not stripped:
                continue
                
            match = pattern.match(stripped)
            if match:
                flush_buffer()
                buffered_prefix, buffered_start_str, buffered_end_str, text_part = match.groups()
                buffered_text = text_part
                buffered_duration = parse_time(buffered_end_str) - parse_time(buffered_start_str)
            else:
                if buffered_prefix is not None:
                    # Append continuation line
                    buffered_text += " " + stripped
                else:
                    # Header/unrecognized line before any timestamps begin
                    outfile.write(line + "\n" if not line.endswith("\n") else line)
                    
        # Flush the final block
        flush_buffer()

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
