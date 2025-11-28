from langchain_huggingface import HuggingFacePipeline
from agents.llm_utils import get_llm
from langchain_core.prompts import PromptTemplate

prettify_llm = get_llm(model="nano", provider="openai", temperature=0.0)
#HuggingFacePipeline.from_model_id(
#    model_id="microsoft/Phi-3-mini-4k-instruct",  # 3.8B Phi-3 Mini instruct model
#    task="text-generation",
#    pipeline_kwargs={
#        "max_new_tokens": 512,
#        "temperature": 0.1,
#        "return_full_text": False,  # only get the completion, not the prompt+completion
#    },
#)

prettify_prompt = PromptTemplate.from_template(
    """You are a Markdown formatter.

Rewrite the text below as clean, well-structured Markdown:
- Add headings where it makes sense.
- Use bullet or numbered lists when appropriate.
- Use code fences for code or commands.
- Fix obvious grammar and spacing.
- Use fency icons to highlight important information.
- Output ONLY Markdown, no explanation.
- **IMPORTANT**: Do not add or modify text, only format!.

Text:
{text}
"""
)

prettify_chain = prettify_prompt | prettify_llm

def prettify(text: str)-> str:
    result = prettify_chain.invoke({"text": text})
    return result.content