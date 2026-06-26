import re

path = 'd:\\WhisperX\\xttsv2_speech_synthesis.py'
with open(path, 'r', encoding='utf-8') as f:
    code = f.read()

# Replace all logging. methods with logger.
code = re.sub(r'logging\.(info|debug|warning|error)\(', r'logger.\1(', code)

# Define logger at top
code = code.replace('import logging', 'import logging\n\nlogger = logging.getLogger("WhisperX")\nlogger.propagate = False')

# Replace the basicConfig setup
old_config = 'logging.basicConfig(level=log_level, format="%(asctime)s - [%(levelname)s] - %(message)s")'
new_config = '''logger.setLevel(log_level)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
        logger.addHandler(ch)'''
code = code.replace(old_config, new_config)

# Remove the force=True line from earlier
old_force = '    # Re-apply our logging configuration because TTS and NeMo modules often hijack the root logger upon import\n    logging.basicConfig(level=log_level, format="%(asctime)s - [%(levelname)s] - %(message)s", force=True)\n'
code = code.replace(old_force, '')

with open(path, 'w', encoding='utf-8') as f:
    f.write(code)
print("Done!")
