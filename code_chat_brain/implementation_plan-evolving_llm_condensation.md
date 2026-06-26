# Auto-Abstract & Prompt Evolution Plan

This plan introduces the `--auto-abstract` feature flag to generate a global context summary before processing sentences, and completely rewrites the system and user prompts to follow the structural insights from ChatGPT, explicitly using English prompts for maximum LLM layer performance as requested.

## Proposed Changes

### `condense_transcription.py`

#### 1. Argparse Addition
Add the `--auto-abstract` flag:
```python
parser.add_argument("--auto-abstract", type=int, default=0, help="Number of lines to read at the start to generate a 2-sentence global context abstract. Use -1 for the entire file (Warning: may exceed context window). Default is 0 (disabled).")
```

#### 2. Abstract Generation Function
Create a new function `generate_abstract(client, model, lines, temperature)`:
- Reads the specified number of lines, strips timestamps, and asks the LLM to write a 2-sentence summary of the video's topic.
- This abstract will be injected into the system prompt for all subsequent sentence condensations.

#### 3. Prompt Rewrites (English only, structured logic)
Rewrite `condense_text` to use highly structured English prompts.

**If `--maintain-context` (Natural Rewrite / Dubbing Adapter):**
> **System Prompt:**
> You are a professional Brazilian Portuguese dubbing adapter.
> {Inject Abstract Here if available: GLOBAL VIDEO CONTEXT: ...}
> Your task is to rewrite a Portuguese subtitle so it sounds natural in spoken Brazilian Portuguese and fits strictly within the allotted time constraint.
> 
> Priorities (highest to lowest):
> 1. Preserve the original meaning.
> 2. Preserve the original tone.
> 3. Make it sound like native speech.
> 4. Stay within the maximum length.
> 
> Rules:
> - If necessary, sacrifice detail instead of meaning.
> - Prefer shorter and more common words.
> - Remove redundancy and merge ideas naturally.
> - Do not add information or explain.
> - OUTPUT STRICTLY IN BRAZILIAN PORTUGUESE.
> - Output ONLY the rewritten subtitle. Do not output conversational filler or quotes.

**If normal (Max Condensation):**
> **System Prompt:**
> You are a professional subtitle condenser for Brazilian Portuguese dubbing focused on extreme synthesis and speech speed.
> {Inject Abstract Here if available: GLOBAL VIDEO CONTEXT: ...}
> Your task is to drastically reduce the Portuguese subtitle to make it as short and direct as possible without losing the core meaning. 
> 
> Rules:
> - Every word must justify its existence.
> - Remove adjectives, filler words, and repeated ideas first.
> - Prefer active voice and common vocabulary.
> - Keep the sentence easy to pronounce aloud.
> - OUTPUT STRICTLY IN BRAZILIAN PORTUGUESE.
> - Output ONLY the condensed subtitle. Do not output conversational filler or quotes.

**User Prompt (Structured format):**
```text
ORIGINAL TEXT:
{original_text}

ABSOLUTE MAXIMUM LIMIT: {int(target_chars)} characters.

Rewrite the text strictly respecting the limit.
```

#### 4. Pre-processing in `main()`
If `--auto-abstract` is `> 0` or `-1`:
- Open the `.txt` file before the main loop.
- Extract the text payload using the existing regex.
- Grab the first `N` lines (or all).
- Call `generate_abstract()` to get the global context string.
- Pass this string to `condense_text` during the main loop.

## Open Questions
> [!IMPORTANT]
> The plan is ready for execution. Please review the English translations of the prompts above. If everything looks good, click **Proceed** and I will implement this logic.
