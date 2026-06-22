import os
import sys
import re
import logging
import argparse
from pathlib import Path

# import ctranslate2
# from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate a transcription file using CTranslate2 and an NLLB model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("transcript_file", nargs="?", help="Path to the original transcription text file")
    parser.add_argument("--model-dir", help="Path to the local CTranslate2 model directory (must contain tokenizer files)")
    parser.add_argument("--source-lang", default="eng_Latn", help="Source language code (FLORES-200 format, e.g. eng_Latn)")
    parser.add_argument("--target-lang", default="por_Latn", help="Target language code (FLORES-200 format, e.g. por_Latn)")
    parser.add_argument("--output-file", "-o", help="Output file path (defaults to <original>_<target_lang>.txt)")
    parser.add_argument("--compute-type", default="int8", choices=["int8", "int8_float16", "int16", "float16", "float32"], help="Compute type for CTranslate2")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run translation on")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for translation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--list-languages", action="store_true", help="List common supported language codes and exit")
    return parser.parse_args()

def translate_transcription():
    args = parse_args()

    if args.list_languages:
        print("==================================================")
        print(" Common NLLB (FLORES-200) Language Codes")
        print("==================================================")
        common_langs = {
            "English": "eng_Latn",
            "Portuguese": "por_Latn",
            "Spanish": "spa_Latn",
            "French": "fra_Latn",
            "German": "deu_Latn",
            "Italian": "ita_Latn",
            "Japanese": "jpn_Jpan",
            "Korean": "kor_Hang",
            "Chinese (Simp)": "zho_Hans",
            "Chinese (Trad)": "zho_Hant",
            "Russian": "rus_Cyrl",
            "Arabic": "arb_Arab",
            "Hindi": "hin_Deva",
            "Dutch": "nld_Latn",
            "Turkish": "tur_Latn",
            "Polish": "pol_Latn",
            "Vietnamese": "vie_Latn",
            "Indonesian": "ind_Latn",
            "Thai": "tha_Thai",
            "Hebrew": "heb_Hebr",
        }
        for name, code in common_langs.items():
            print(f"  {name:<20} : {code}")
        print("\n* NLLB supports over 200 languages!")
        print("* For the complete list of FLORES-200 codes, visit:")
        print("* https://github.com/facebookresearch/flores/blob/main/flores200/README.md")
        print("==================================================")
        sys.exit(0)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - [%(levelname)s] - %(message)s")

    if not args.transcript_file:
        logging.error("The following arguments are required: transcript_file")
        sys.exit(1)
        
    if not args.model_dir:
        logging.error("The following arguments are required: --model-dir")
        sys.exit(1)

    transcript_path = Path(args.transcript_file)
    if not transcript_path.is_file():
        logging.error(f"Transcription file not found: {args.transcript_file}")
        sys.exit(1)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logging.error(f"Model directory not found: {args.model_dir}")
        sys.exit(1)

    output_path = args.output_file
    if not output_path:
        output_path = transcript_path.with_name(f"{transcript_path.stem}_{args.target_lang}{transcript_path.suffix}")
    else:
        output_path = Path(output_path)

    # 1. Load Tokenizer
    logging.debug("Loading transformers library...")
    from transformers import AutoTokenizer
    logging.info(f"Loading Tokenizer from {model_dir}...")
    try:
        # NLLB uses the base architecture vocabulary. We load it locally from the downloaded directory.
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), src_lang=args.source_lang)
    except Exception as e:
        logging.error(f"Failed to load tokenizer from {model_dir}: {e}")
        logging.info("Make sure the model directory contains tokenizer configuration files (tokenizer.json, config.json, etc.).")
        sys.exit(1)

    # 2. Load Translation Model
    logging.debug("Loading ctranslate2 library...");
    import ctranslate2
    logging.info(f"Loading CTranslate2 Model from {model_dir} onto {args.device.upper()} (compute_type: {args.compute_type})...")
    try:
        translator = ctranslate2.Translator(
            str(model_dir),
            device=args.device,
            device_index=0 if args.device == "cuda" else 0,
            compute_type=args.compute_type
        )
    except Exception as e:
        logging.error(f"Failed to load CTranslate2 model: {e}")
        sys.exit(1)

    # 3. Read and parse transcription file
    logging.info(f"Reading transcription from {transcript_path}")
    lines_data = []
    
    # Regex to capture the timestamp and speaker prefix, e.g., "[041.94s - 055.41s] Speaker SPEAKER_01: "
    prefix_pattern = re.compile(r"^(\[[^\]]+\](?:.*?:)?\s*)(.*)$")

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                lines_data.append(("", ""))
                continue
            
            match = prefix_pattern.match(line)
            if match:
                prefix = match.group(1)
                text = match.group(2)
            else:
                prefix = ""
                text = line
            
            lines_data.append((prefix, text))

    texts_to_translate = [text for prefix, text in lines_data if text.strip()]
    logging.info(f"Found {len(texts_to_translate)} non-empty lines to translate.")

    if not texts_to_translate:
        logging.warning("No text found to translate.")
        sys.exit(0)

    # 4. Tokenize and Translate in Batches
    logging.info("Tokenizing text...")
    source_tokens = []
    for text in texts_to_translate:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        source_tokens.append(tokens)

    logging.info(f"Translating in batches of {args.batch_size}...")
    translated_texts = []
    
    # Perform translation
    try:
        results = translator.translate_batch(
            source_tokens,
            target_prefix=[[args.target_lang]] * len(source_tokens),
            max_batch_size=args.batch_size,
            beam_size=4,
            max_decoding_length=256
        )
        
        for idx, res in enumerate(results):
            target_tokens = res.hypotheses[0]
            # Remove the target language token if it is generated as the first token
            if target_tokens and target_tokens[0] == args.target_lang:
                target_tokens = target_tokens[1:]
            
            decoded_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens))
            translated_texts.append(decoded_text)
            
            if args.verbose and idx < 5:
                logging.debug(f"Sample {idx+1}: {decoded_text}")
                
    except Exception as e:
        logging.error(f"Translation failed during batch processing: {e}")
        sys.exit(1)

    # 5. Reconstruct and Save
    logging.info(f"Saving translated transcription to {output_path}")
    translated_idx = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for prefix, text in lines_data:
            if text.strip():
                translated_text = translated_texts[translated_idx]
                f.write(f"{prefix}{translated_text}\n")
                translated_idx += 1
            else:
                f.write(f"{prefix}\n")

    logging.info("Translation completed successfully!")
    print("\n==================================================")
    print("Translation Summary:")
    print(f"  Source File       : {transcript_path.name}")
    print(f"  Model Path        : {model_dir}")
    print(f"  Languages         : {args.source_lang} -> {args.target_lang}")
    print(f"  Lines Translated  : {len(translated_texts)}")
    print(f"  Output File       : {output_path.resolve()}")
    print("==================================================")

if __name__ == "__main__":
    translate_transcription()