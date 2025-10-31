# summarizer.py
from langchain_openai import ChatOpenAI
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# You can adjust the model name/temperature as needed.
model = ChatOpenAI(temperature=0, model="gpt-4.1-nano")

# Define a simple summarization prompt template.
_TEMPLATE = """Here is a document excerpt:

{context}

Please provide a detailed summary:"""

prompt = ChatPromptTemplate.from_template(_TEMPLATE)
chain = prompt | model | StrOutputParser()

def summarize_text(context: str) -> str:
    """
    Given a block of text, invoke the LLM chain to produce a detailed summary.
    """
    return chain.invoke({"context": context})
