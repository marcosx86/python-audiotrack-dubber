# Implementation Walkthrough: LLM Text Condensation

We have officially built the script to harness your local LM Studio instance for intelligent dubbing condensation!

## The Script: `condense_transcription.py`
This is a brand new standalone script designed to sit perfectly between your translation step and your audio synthesis step. 

### How it works:
1. It reads your translated `.txt` file and extracts the exact `[00:00:00.000 --> 00:00:05.000]` timestamps.
2. It calculates the strict **target duration** for that sentence.
3. It multiplies the duration by the `--max-chars-per-sec` (default: 15) to find the absolute maximum length the Portuguese text can be before it causes timeline drift or chipmunk-speed.
4. **If the text is too long:** It silently pings your LM Studio local server via the `openai` python package.
5. It uses a very strict System Prompt with a `0.1` temperature, commanding your local LLM to rewrite the Portuguese text to be punchier and shorter while maintaining the original meaning. It strictly forbids conversational "garbage" like intros or quotes.
6. It saves the results perfectly back into a new `.txt` file with the exact original timestamp formatting so it can be fed straight into `speech_synthesis.py`!

### Security & Parameters
- By default, it connects to LM Studio on `http://localhost:1234/v1`.
- For API Key security, you can either pass `--api-key YOUR_KEY`, or it will securely check for an environment variable called `LM_STUDIO_API_KEY`. If neither exists, it just sends `"lm-studio"` (which is what local LM Studio expects by default anyway!).

### Usage Example
```bash
# Export your secure token if your LM Studio requires it
set LM_STUDIO_API_KEY=my_secure_token

# Run the condensation pass
python condense_transcription.py why_it_sucks_to_be_a_vestal_virgin_in_anscient_rome_por_Latn.txt --output-file condensed_output.txt --verbose
```

At the end of the script, it will print a beautiful log showing exactly how many lines it condensed and exactly how many characters it managed to shave off your script!
