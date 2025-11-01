# ingestion/loader.py
import logging, sys, os, chardet
from json import JSONDecodeError
from pathlib import Path

from langchain_classic.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import (
    TextLoader, 
    UnstructuredPDFLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader,
    AssemblyAIAudioTranscriptLoader,
    UnstructuredCSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredXMLLoader,
    UnstructuredRTFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredEmailLoader,
    JSONLoader,
)

try:
    from pydub import AudioSegment
except ModuleNotFoundError as exc:
    AudioSegment = None
    _pydub_import_error = exc
else:
    _pydub_import_error = None
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/loader.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

import config

try:
    import pdfminer.utils as _pdfminer_utils
except Exception:  # pragma: no cover - optional dependency failure
    _pdfminer_utils = None
else:
    if _pdfminer_utils is not None and not hasattr(_pdfminer_utils, "open_filename"):
        def _open_filename(filename, *args, **kwargs):
            """Compatibility shim for pdfminer >=202407 which removed open_filename."""
            if hasattr(filename, "read"):
                return filename
            filename = os.fspath(filename)
            mode = args[0] if args else kwargs.get("mode")
            if mode is None:
                kwargs.setdefault("mode", "rb")
                mode = kwargs["mode"]
            if filename == "-":
                return sys.stdin.buffer if "b" in mode else sys.stdin
            return open(filename, *args, **kwargs)

        _pdfminer_utils.open_filename = _open_filename


def _ensure_audio_support() -> None:
    if AudioSegment is None:
        missing = getattr(_pydub_import_error, "name", "pydub")
        raise RuntimeError(
            "Audio ingestion requires optional dependency `pydub` "
            "with an `audioop` backend. Install it via `pip install pydub audioop-lts`."
        ) from _pydub_import_error


def _log_missing_audio_dependency(filename: str) -> None:
    missing = getattr(_pydub_import_error, "name", "pydub")
    logging.warning(
        f"Skipping audio file {filename} because optional dependency {missing!r} is not installed. "
        "Install `pip install pydub audioop-lts` to enable audio ingestion."
    )


def convert_audio_to_wav(input_file, output_file, audio_type):
    """
    Converts an M4A (or similar) file to WAV format.
    """
    _ensure_audio_support()
    audio = AudioSegment.from_file(input_file, format=audio_type)
    audio.export(output_file, format='wav')


def get_assebmblyai_loader(filename, full_path):
    wav_name = filename
    name, ext = os.path.splitext(filename)
    ext = ext.replace('.', '')
    wav_name = f"temp/{name}.wav"

    convert_audio_to_wav(full_path, wav_name, ext)
    import assemblyai
    config = assemblyai.TranscriptionConfig(
        language_code='ru'
    )
    return AssemblyAIAudioTranscriptLoader(wav_name, config=config)

def get_loader(root: str, filename: str) -> BaseLoader:

    full_path = os.path.join(root, filename)
    ext = os.path.splitext(filename)[1].lower()
    loader = None

    if ext in ['.txt', '.py']:
        loader = TextLoader(full_path, encoding="utf-8")
    elif ext == '.json':
        loader = JSONLoader(full_path, jq_schema=".", text_content=False)
    elif ext == '.pdf':
        loader = UnstructuredPDFLoader(full_path, mode="single", strategy="hi_res", infer_table_structure=True, languages=["ru", "en"])
    elif ext in ['.docx', '.doc']:
        loader = UnstructuredWordDocumentLoader(full_path, mode="single", strategy="hi_res", infer_table_structure=True)
    elif ext in ['.pptx', '.ppt']:
        # PowerPoint loader: returns a single Document by default
        loader = UnstructuredPowerPointLoader(full_path, mode="single")
    elif ext in ['.xlsx', '.xls']:
        # Excel loader: returns a single Document by default
        loader = UnstructuredExcelLoader(full_path, mode="elements", find_subtable=False)
    elif ext in ['.csv']:
        loader = UnstructuredCSVLoader(full_path, mode="elements")
    elif ext in ['.html']:
        loader = UnstructuredHTMLLoader(full_path, mode="single")
    elif ext in ['.xml']:
        loader = UnstructuredXMLLoader(full_path, mode="single")
    elif ext in ['.rtf']:
        loader = UnstructuredRTFLoader(full_path, mode="single")
    elif ext in ['.md']:
        loader = UnstructuredMarkdownLoader(full_path, mode="single")
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
        loader = UnstructuredImageLoader(full_path, mode="single", strategy="hi_res", infer_table_structure=True, languages=["ru", "en"])
    elif ext in ['.msg', '.eml']:
        loader = UnstructuredEmailLoader(full_path, mode="single", process_attachments=True, strategy="hi_res", infer_table_structure=True, languages=["ru", "en"])
    elif ext in ['.mp4', '.mov', '.avi']:
        if AudioSegment is None:
            _log_missing_audio_dependency(filename)
        else:
            loader = get_assebmblyai_loader(filename, full_path)
    return loader



def fallback_json(full_path: str)-> list[Document]:
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

def fallback_text(full_path: str)-> list[Document]:
    with open(full_path, "rb") as f:
        raw_data = f.read(10000)  # read a chunk, not whole file
    result = chardet.detect(raw_data)
    if not result or "encoding" not in result:
        return None
    encoding = result["encoding"]
    loader = TextLoader(full_path, encoding=encoding)
    return loader.load()


def load_single_document(file_path: str) -> list[Document]:
    """
    Load a single document from the filesystem, applying the same fallbacks
    and metadata enrichment as load_documents() but limited to a given file.
    """
    directory_path = os.path.dirname(file_path)
    filename = os.path.basename(file_path)

    loader = get_loader(directory_path, filename)
    if loader is None:
        logging.warning("Unsupported file type %s for file %s", os.path.splitext(filename)[1].lower(), filename)
        return []

    try:
        docs = loader.load()
    except Exception as e:
        docs = None
        if isinstance(loader, JSONLoader) and isinstance(e, JSONDecodeError):
            docs = fallback_json(file_path)
        elif isinstance(loader, TextLoader) and isinstance(e, RuntimeError):
            docs = fallback_text(file_path)
        if docs is None:
            raise e

    rel_path = os.path.relpath(file_path, directory_path)
    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["relative_path"] = rel_path
    return docs


def load_documents(directory_path: str, extentions: list[str] = None) -> list[Document]:
    """
    Recursively scans the given directory, loads supported files, and returns a list of LangChain Documents.
    
    Supported file types:
      - .txt, .md     : Loaded using TextLoader
      - .pdf          : Loaded using UnstructuredPDFLoader
      - .doc, .docx   : Loaded using UnstructuredWordDocumentLoader
    
    Each document gets assigned metadata including:
       - source: file name
       - relative_path: file path relative to the base directory
    
    Args:
        directory_path (str): The path to the base directory containing documents.
    
    Returns:
        List[Document]: List of loaded LangChain Document objects.
    """
    documents = []
    # Walk through directories recursively
    for root, _, files in os.walk(directory_path):
        for filename in files:
            full_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            if extentions and ext not in extentions:
                continue
            logging.info(f"Processing {full_path}...")
            try:
                loader = get_loader(root, filename)
                if loader is None:
                    logging.warning(f"Unsupported file type {ext} for file {filename}")
                    continue
                try:
                    docs = loader.load()
                except Exception as e:
                    docs = None
                    if isinstance(loader, JSONLoader) and isinstance(e, JSONDecodeError):
                        docs = fallback_json(full_path)
                    elif isinstance(loader, TextLoader) and isinstance(e, RuntimeError):
                        docs = fallback_text(full_path)
                    if docs is None:
                        raise e
                # Compute relative path from the base directory
                rel_path = os.path.relpath(full_path, directory_path)
                # Set metadata for each loaded document
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["relative_path"] = rel_path
                documents.extend(docs)
                logging.info(f"...{full_path} processed.")
            except Exception as e:
                logging.error(f"Error loading file {filename}: {e}")
    return documents


if __name__ == "__main__":
    import pickle

    DOCUMENT_DIR = "data/itil"
    documents = []
    if os.path.isfile("data/documents/itil_docstore.pkl"):
        with open('data/documents/itil_docstore.pkl', 'rb') as file:
            documents = pickle.load(file)
    newdocs = load_documents(DOCUMENT_DIR) #, extentions=[".txt"])
    documents.extend(newdocs)
    with open('data/documents/itil_docstore.pkl', 'wb') as file:
        pickle.dump(documents, file)

    print(f"Loaded {len(documents)} documents.")
