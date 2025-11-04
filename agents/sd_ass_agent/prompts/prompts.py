from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent

"""
with (_PROMPTS_DIR / "working_prompt_sales.txt").open(encoding="utf-8") as f:
    sm_prompt = f.read()
with (_PROMPTS_DIR / "working_prompt.txt").open(encoding="utf-8") as f:
    sd_prompt = f.read()
with (_PROMPTS_DIR / "working_prompt_employee.txt").open(encoding="utf-8") as f:
    default_prompt = f.read()
with (_PROMPTS_DIR / "supervisor_prompt.txt").open(encoding="utf-8") as f:
    sv_prompt = f.read()

with (_PROMPTS_DIR / "search_web_prompt.txt").open(encoding="utf-8") as f:
    sd_agent_web_prompt = f.read()
with (_PROMPTS_DIR / "search_web_prompt_sales.txt").open(encoding="utf-8") as f:
    sm_agent_web_prompt = f.read()
with (_PROMPTS_DIR / "search_web_prompt_employee.txt").open(encoding="utf-8") as f:
    default_search_web_prompt = f.read()
"""

with (_PROMPTS_DIR / "working_prompt_en.txt").open(encoding="utf-8") as f:
    sv_prompt = default_prompt = sd_prompt = sm_prompt = f.read()

with (_PROMPTS_DIR / "search_web_prompt_en.txt").open(encoding="utf-8") as f:
    default_search_web_prompt = sm_agent_web_prompt = sd_agent_web_prompt = f.read()
