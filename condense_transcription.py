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
    parser.add_argument("--chat-mode", action="store_true", help="Maintain a continuous chat history with the LLM across the entire script for full context. Warning: Uses more VRAM and slows down over time.")
    parser.add_argument("--auto-abstract", type=int, default=0, help="Number of lines to read at the start to generate a 2-sentence global context abstract. Use -1 for the entire file. Default is 0 (disabled).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()

def parse_time(time_str):
    if ':' in time_str:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        return float(time_str.replace('s', ''))

def generate_abstract(client, model, abstract_text, temperature=0.1):
    system_prompt = "You are an expert summarizer. Read the following video subtitle script and write a 2-sentence summary of the video's core topic and context. Do not include introductory phrases. Output strictly the summary."
    user_prompt = f"SCRIPT EXTRACT:\n{abstract_text}"
    
    try:
        logging.debug("Sending abstract generation request to LLM...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=2048,
        )
        abstract = response.choices[0].message.content.strip()
        # Automatically strip reasoning blocks if a DeepSeek R1 model is used
        abstract = re.sub(r'<think>.*?</think>', '', abstract, flags=re.DOTALL).strip()
        
        logging.info(f"Generated Global Abstract: {abstract}")
        return abstract
    except Exception as e:
        logging.error(f"Failed to generate abstract: {e}")
        return None

def condense_text(client, model, original_text, target_chars, temperature=0.1, maintain_context=False, chat_history=None, abstract=None):
    abstract_injection = f"\nGLOBAL VIDEO CONTEXT: {abstract}\n" if abstract else ""
    
    if maintain_context:
        system_prompt = (
            "You are a professional Brazilian Portuguese dubbing adapter.\n"
            f"{abstract_injection}"
            "Your task is to rewrite a Portuguese subtitle so it sounds natural in spoken Brazilian Portuguese and fits strictly within the allotted time constraint.\n\n"
            "Priorities (highest to lowest):\n"
            "1. Preserve the original meaning.\n"
            "2. Preserve the original tone.\n"
            "3. Make it sound like native speech.\n"
            "4. Stay within the maximum length.\n\n"
            "Rules:\n"
            "- If necessary, sacrifice detail instead of meaning.\n"
            "- Prefer shorter and more common words.\n"
            "- Remove redundancy and merge ideas naturally.\n"
            "- Do not add information or explain.\n"
            "- DO NOT output reasoning, thinking, or <think> blocks.\n"
            "- OUTPUT STRICTLY IN BRAZILIAN PORTUGUESE.\n"
            "- Output ONLY the rewritten subtitle. Do not output conversational filler or quotes."
        )
    else:
        system_prompt = (
            "You are a professional subtitle condenser for Brazilian Portuguese dubbing focused on extreme synthesis and speech speed.\n"
            f"{abstract_injection}"
            "Your task is to drastically reduce the Portuguese subtitle to make it as short and direct as possible without losing the core meaning.\n\n"
            "Rules:\n"
            "- Every word must justify its existence.\n"
            "- Remove adjectives, filler words, and repeated ideas first.\n"
            "- Prefer active voice and common vocabulary.\n"
            "- Keep the sentence easy to pronounce aloud.\n"
            "- DO NOT output reasoning, thinking, or <think> blocks.\n"
            "- OUTPUT STRICTLY IN BRAZILIAN PORTUGUESE.\n"
            "- Output ONLY the condensed subtitle. Do not output conversational filler or quotes."
        )
        
    user_prompt = (
        "ORIGINAL TEXT:\n"
        f"{original_text}\n\n"
        f"ABSOLUTE MAXIMUM LIMIT: {int(target_chars)} characters.\n\n"
        "Rewrite the text strictly respecting the limit."
    )
    
    if chat_history is not None:
        if not chat_history:
            chat_history.append({"role": "system", "content": system_prompt})
        chat_history.append({"role": "user", "content": user_prompt})
        messages = chat_history
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
    try:
        logging.debug(f"Sending request to LLM...")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
        )
        condensed = response.choices[0].message.content.strip()
        
        # Strip reasoning blocks if the model ignores the instruction
        condensed = re.sub(r'<think>.*?</think>', '', condensed, flags=re.DOTALL).strip()
        
        # Clean up accidental quotes if the model added them
        if condensed.startswith('"') and condensed.endswith('"'):
            condensed = condensed[1:-1]
            
        if chat_history is not None:
            chat_history.append({"role": "assistant", "content": condensed})
            
        logging.debug(f"Condensed text from LLM: {condensed}")
        return condensed
    except Exception as e:
        logging.error(f"Failed to condense text: {e}")
        # On failure in chat mode, remove the user message so it doesn't pollute context
        if chat_history is not None and len(chat_history) > 0 and chat_history[-1]["role"] == "user":
            chat_history.pop()
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
        
        # Trigger JIT load into VRAM with explicit context window size.
        # This ensures the model is loaded with the exact memory footprint requested
        # before we start the condensation timing loop.
        logging.info(f"Triggering JIT load for model '{args.model}' with context length {args.context_length}...")
        client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "warmup"}],
            max_tokens=1,
            extra_body={
                "context_length": args.context_length,
                "n_ctx": args.context_length
            }
        )
        logging.info(f"Model '{args.model}' successfully loaded into VRAM and ready.")
            
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
    
    global_abstract = None
    if args.auto_abstract != 0:
        logging.info("Extracting text for global abstract generation...")
        abstract_lines = []
        with open(args.translated_transcription, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if args.auto_abstract > 0 and i >= args.auto_abstract:
                    break
                match = pattern.match(line)
                if match:
                    abstract_lines.append(match.group(4).strip())
                elif line.strip():
                    abstract_lines.append(line.strip())
        
        if abstract_lines:
            abstract_text = "\n".join(abstract_lines)
            global_abstract = generate_abstract(client, args.model, abstract_text, args.temperature)
        else:
            logging.warning("Failed to extract any text for abstract generation.")
    
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
        
        # Initialize conversation history if chat mode is enabled
        conversation_history = [] if args.chat_mode else None
        
        def flush_buffer():
            nonlocal lines_processed, condensed_count, total_chars_saved
            if buffered_prefix is None:
                return
                
            text = buffered_text.strip()
            target_char_limit = buffered_duration * args.max_chars_per_sec
            
            if buffered_duration > 0 and len(text) > target_char_limit:
                logging.debug(f"[{buffered_start_str} - {buffered_end_str}] Too long ({len(text)} > {int(target_char_limit)}). Condensing...")
                
                llm_t0 = time.time()
                condensed = condense_text(client, args.model, text, target_char_limit, args.temperature, args.maintain_context, conversation_history, global_abstract)
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
