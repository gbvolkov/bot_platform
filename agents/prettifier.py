from langchain_huggingface import HuggingFacePipeline
from agents.llm_utils import get_llm, with_llm_fallbacks
from langchain_core.prompts import PromptTemplate

import config
import logging

_prettify_primary_llm = get_llm(model="mini", provider="openai", temperature=0.0)
_prettify_alternative_llm = get_llm(model="base", provider="openai", temperature=0.0)
prettify_llm = with_llm_fallbacks(
    _prettify_primary_llm,
    alternative_llm=_prettify_alternative_llm,
    primary_retries=3,
)
#HuggingFacePipeline.from_model_id(
#    model_id="microsoft/Phi-3-mini-4k-instruct",  # 3.8B Phi-3 Mini instruct model
#    task="text-generation",
#    pipeline_kwargs={
#        "max_new_tokens": 512,
#        "temperature": 0.1,
#        "return_full_text": False,  # only get the completion, not the prompt+completion
#    },
#)

#prettify_prompt = PromptTemplate.from_template(
#    """You are a Markdown formatter.
#
#Rewrite the text below as clean, well-structured Markdown:
#- Add headings where it makes sense.
#- Use bullet or numbered lists when appropriate.
#- Use code fences for code or commands.
#- **IMPORTANT** Format links properly! Pay attention to titles!
#- **IMPORTANT** Do not change wording!
#- **IMPORTANT** Do not remove or cut any information!
#- Fix obvious grammar and spacing.
#- Use fency icons to highlight important information.
#- Output ONLY Markdown, no explanation.
#- **IMPORTANT**: Do not add or modify text, only format!.
#
#Text:
#{text}
#"""
#)

prettify_prompt = PromptTemplate.from_template(
    """You are experienced and accurate text editor.

Your are receiving an artifact text thet shall be cleaned up and formatted in MarkdownV2.
Please remove from text all dirty texts, discussions fwith user, questions and so on and leave the rest of the text intact.

-----------------------------------------------------------------------------------------
Examples of text which shall be removed:
1) "Хотите загрузить реальные данные (интервью, таблицы по TAT/rework/adoption, отчёты/претензии/ИБ-опросники) или создаём вручную? "
2) "Подтверждаете артефакт 8 без изменений? Напишите «подтверждаю».
AS-IS/TO-BE: ВЫСОКО — обе части описаны и сопоставимы.
Входы/выходы: ВЫСОКО — чёткие границы процесса.
Владельцы: ВЫСОКО — назначены владельцы процесса/SLA/качества/интеграции.
Узкие места: ВЫСОКО — перечислены + привязаны гипотезы улучшений."
-----------------------------------------------------------------------------------------

Additionally format resulting text as MarkdownV2:
- Add headings where it makes sense.
- Use bullet or numbered lists when appropriate.
- Use code fences for code or commands.
- **IMPORTANT** Format links properly! Pay attention to titles!
- **IMPORTANT** Do not change wording!
- **IMPORTANT** Do not remove or cut any information!
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
    #return text
    try:
        result = prettify_chain.invoke({"text": text})
    except Exception as e:
        logging.error(f"Error occured during prettify tool calling.\nException: {e}")
        return text
    return result.content
