import pandas as pd
from langchain_core.documents import Document  # LangChain Document class

def get_documents_for_sd_qa(df: pd.DataFrame) -> list[Document]:
    # Create a list of Documents from each row
    documents = []
    for _, row in df.iterrows():
        # Combine relevant fields into the page_content for retrieval
        content = (
            f"It System: {row['it_system']}\n"
            f"Problem Description: {row['problem_description']}\n"
            f"Problem Solution: {row['problem_solution']}\n\n"
            f"Ссылка на статью:https://kb.ileasing.ru/space/{row['space_id']}/article/{row['article_id']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "it_system": row["it_system"],
                "problem_description": row["problem_description"],
                "problem_solution": row["problem_solution"],
                "space_id": row["space_id"],
                "article_id": row["article_id"],
                "article_title": row["article_title"],
                "source": "sd_qa_table"
            }
        )
        documents.append(doc)
    return documents

def get_documents_for_sd_tickets(df: pd.DataFrame) -> list[Document]:
    # Create a list of Documents from each row
    documents = []
    for _, row in df.iterrows():
        # Combine relevant fields into the page_content for retrieval
        content = (
            f"Ticket number: {row["Тикет"]}\n"
            f"Ticket topic: {row["Тема"]}\n"
            f"Problem: {row["Описание текст"]}\n\n"
            f"Solution: {row["Описание решения"]}\n\n"
            f"Ссылка на тикет: https://kb.ileasing.ru/space/{row['st_id']}/article/{row['id']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "ticket_no": row["Тикет"],
                "topic": row["Тема"],
                "ticket_dt": row["Дата регистрации"],
                "ticket_type": row["Критерий ошибки"],
                "problem": row["Описание текст"],
                "solution": row["Описание решения"],
                "space_id": row["st_id"],
                "article_id": row["id"],
                "article_title": row["Тема"],
                "source": "sd_tickets_table"
            }
        )
        documents.append(doc)
    return documents

def get_documents_for_glossary(df: pd.DataFrame) -> list[Document]:
    # Create a list of Documents from each row
    documents = []
    for _, row in df.iterrows():
        # Combine relevant fields into the page_content for retrieval
        content = (
            f"Term: {row['term']}\n"
            f"Definition: {row['definition']}\n"
            f"Ссылка на статью:https://kb.ileasing.ru/space/{row['space_id']}/article/{row['article_id']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "term": row["term"],
                "definition": row["definition"],
                "space_id": row["space_id"],
                "article_id": row["article_id"],
                "article_title": row["article_title"],
                "source": row["section"]
            }
        )
        documents.append(doc)
    return documents