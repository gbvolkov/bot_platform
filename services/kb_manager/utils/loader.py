# ingestion/loader.py
import logging
import os
import sys
from json import JSONDecodeError
from pathlib import Path

import chardet
from langchain_classic.docstore.document import Document
from langchain_community.document_loaders import (
    AssemblyAIAudioTranscriptLoader,
    JSONLoader,
    TextLoader,
    UnstructuredCSVLoader,
    UnstructuredEmailLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredImageLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRTFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredXMLLoader,
)
from langchain_community.document_loaders.base import BaseLoader

try:
    from pydub import AudioSegment
except Exception:  # pragma: no cover - optional dependency
    AudioSegment = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/loader.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

import config


def _ensure_audio_support() -> None:
    if AudioSegment is None:
        raise RuntimeError(
            "Audio ingestion requires optional dependency `pydub` with a working audio backend. "
            "Install `pydub` and an audio backend such as `audioop-lts`."
        )


def convert_audio_to_wav(input_file, output_file, audio_type):
    """Converts an M4A (or similar) file to WAV format."""
    _ensure_audio_support()
    audio = AudioSegment.from_file(input_file, format=audio_type)
    audio.export(output_file, format="wav")


def get_assebmblyai_loader(filename, full_path):
    wav_name = filename
    name, ext = os.path.splitext(filename)
    ext = ext.replace(".", "")
    wav_name = f"temp/{name}.wav"

    convert_audio_to_wav(full_path, wav_name, ext)
    import assemblyai

    transcription_config = assemblyai.TranscriptionConfig(language_code="ru")
    return AssemblyAIAudioTranscriptLoader(wav_name, config=transcription_config)


def get_loader(root: str, filename: str) -> BaseLoader:
    full_path = os.path.join(root, filename)
    ext = os.path.splitext(filename)[1].lower()
    loader = None

    if ext in [".txt", ".py"]:
        loader = TextLoader(full_path, encoding="utf-8")
    elif ext == ".json":
        loader = JSONLoader(full_path, jq_schema=".", text_content=False)
    elif ext == ".pdf":
        loader = UnstructuredPDFLoader(full_path, mode="single", strategy="hi_res", infer_table_structure=True, languages=["ru", "en"])
    elif ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(full_path, mode="single", strategy="hi_res", infer_table_structure=True)
    elif ext in [".pptx", ".ppt"]:
        loader = UnstructuredPowerPointLoader(full_path, mode="single")
    elif ext in [".xlsx", ".xls"]:
        loader = UnstructuredExcelLoader(full_path, mode="elements", find_subtable=False)
    elif ext in [".csv"]:
        loader = UnstructuredCSVLoader(full_path, mode="elements")
    elif ext in [".html"]:
        loader = UnstructuredHTMLLoader(full_path, mode="single")
    elif ext in [".xml"]:
        loader = UnstructuredXMLLoader(full_path, mode="single")
    elif ext in [".rtf"]:
        loader = UnstructuredRTFLoader(full_path, mode="single")
    elif ext in [".md"]:
        loader = UnstructuredMarkdownLoader(full_path, mode="single")
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
        loader = UnstructuredImageLoader(full_path, mode="single", strategy="hi_res", infer_table_structure=True, languages=["ru", "en"])
    elif ext in [".msg", ".eml"]:
        loader = UnstructuredEmailLoader(full_path, mode="single", process_attachments=True, strategy="hi_res", infer_table_structure=True, languages=["ru", "en"])
    elif ext in [".mp4", ".mov", ".avi"]:
        loader = get_assebmblyai_loader(filename, full_path)
    return loader


def fallback_json(full_path: str) -> list[Document]:
    try:
        src = Path(full_path)
        text = src.read_text(encoding="utf-8-sig")
        tmp = src.with_suffix(".nobom.json")
        tmp.write_text(text, encoding="utf-8")
    except Exception:
        return None
    try:
        loader = JSONLoader(file_path=str(tmp), jq_schema=".", text_content=False)
        docs = loader.load()
    finally:
        tmp.unlink(missing_ok=True)
    return docs


def fallback_text(full_path: str) -> list[Document]:
    with open(full_path, "rb") as file_handle:
        raw_data = file_handle.read(10000)
    result = chardet.detect(raw_data)
    if not result or "encoding" not in result:
        return None
    loader = TextLoader(full_path, encoding=result["encoding"])
    return loader.load()


def load_single_document(file_path: str) -> list[Document]:
    """Load a single document from the filesystem."""
    directory_path = os.path.dirname(file_path)
    filename = os.path.basename(file_path)

    loader = get_loader(directory_path, filename)
    if loader is None:
        logging.warning("Unsupported file type %s for file %s", os.path.splitext(filename)[1].lower(), filename)
        raise NotImplementedError(f"Unsupported file type {os.path.splitext(filename)[1].lower()} for file {filename}")
    try:
        docs = loader.load()
    except Exception as exc:
        docs = None
        if isinstance(loader, JSONLoader) and isinstance(exc, JSONDecodeError):
            docs = fallback_json(file_path)
        elif isinstance(loader, TextLoader) and isinstance(exc, RuntimeError):
            docs = fallback_text(file_path)
        if docs is None:
            raise exc

    rel_path = os.path.relpath(file_path, directory_path)
    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["relative_path"] = rel_path
    return docs


def load_documents(directory_path: str, extentions: list[str] = None) -> list[Document]:
    """Recursively scan the given directory and return LangChain documents."""
    documents = []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            full_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            if extentions and ext not in extentions:
                continue
            logging.info("Processing %s...", full_path)
            try:
                loader = get_loader(root, filename)
                if loader is None:
                    logging.warning("Unsupported file type %s for file %s", ext, filename)
                    continue
                try:
                    docs = loader.load()
                except Exception as exc:
                    docs = None
                    if isinstance(loader, JSONLoader) and isinstance(exc, JSONDecodeError):
                        docs = fallback_json(full_path)
                    elif isinstance(loader, TextLoader) and isinstance(exc, RuntimeError):
                        docs = fallback_text(full_path)
                    if docs is None:
                        raise exc
                rel_path = os.path.relpath(full_path, directory_path)
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["relative_path"] = rel_path
                documents.extend(docs)
                logging.info("...%s processed.", full_path)
            except Exception as exc:
                logging.error("Error loading file %s: %s", filename, exc)
    return documents


if __name__ == "__main__":
    documents = []
    if os.path.isfile("data/documents/itil_docstore.pkl"):
        import pickle

        with open("data/documents/itil_docstore.pkl", "rb") as file_handle:
            documents = pickle.load(file_handle)
    new_docs = load_documents("data/itil")
    documents.extend(new_docs)
    with open("data/documents/itil_docstore.pkl", "wb") as file_handle:
        import pickle

        pickle.dump(documents, file_handle)

    print(f"Loaded {len(documents)} documents.")
