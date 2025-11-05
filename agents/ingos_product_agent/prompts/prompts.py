from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent

with (_PROMPTS_DIR / "working_prompt_en.txt").open(encoding="utf-8") as f:
    product_prompt = f.read()
