from dotenv import load_dotenv,dotenv_values
import os

from pathlib import Path


if os.path.exists("gv.env"):
    load_dotenv('./gv.env')
else:
    documents_path = Path.home() / ".env"
    load_dotenv(os.path.join(documents_path, 'gv.env'))

CHROME_DRIVER_PATH = './bin/chromedriver.exe'
LOGIN_URL = 'https://kb.ileasing.ru/auth/sign-in?redirect=%2Fspace%2Fa100dc8d-3af0-418c-8634-f09f1fdb06f2%2Farticle%2Fa1038bbc-e5d9-4b5a-9482-2739c19cb6cb'
TARGET_URL = 'https://kb.ileasing.ru/space/a100dc8d-3af0-418c-8634-f09f1fdb06f2/article/a1038bbc-e5d9-4b5a-9482-2739c19cb6cb'
USERNAME = '7810155'
PASSWORD = os.environ.get('IL_PWD') 

LOCAL_MODEL_NAME='/models/Meta-Llama-3.1-8B-Instruct-Q8_0'

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_ASSISTANT_BOT_TOKEN')
#EMBEDDING_MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

#EMBEDDING_MODEL='sentence-transformers/distiluse-base-multilingual-cased-v1'
EMBEDDING_MODEL=os.environ.get('EMBEDDING_MODEL') or '/models/multilingual-e5-large'
#EMBEDDING_MODEL='jinaai/jina-embeddings-v3'

GIGA_CHAT_USER_ID=os.environ.get('GIGA_CHAT_USER_ID')
GIGA_CHAT_SECRET = os.environ.get('GIGA_CHAT_SECRET')
GIGA_CHAT_AUTH = os.environ.get('GIGA_CHAT_AUTH')
GIGA_CHAT_SCOPE = "GIGACHAT_API_PERS"

LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY', "")
#LANGCHAIN_API_KEY_DEV = os.environ.get('LANGCHAIN_API_KEY_DEV')
LANGCHAIN_ENDPOINT = "" #"https://api.smith.langchain.com"
USE_ANONIMIZER=(os.environ.get('USE_ANONIMIZER', "False").lower() == 'true')
if USE_ANONIMIZER:
    LANGSMITH_HIDE_INPUTS=True
    LANGSMITH_HIDE_OUTPUTS=True
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"

#os.environ["LANGCHAIN_ENDPOINT"]=LANGCHAIN_ENDPOINT
#os.environ["LANGCHAIN_TRACING_V2"] = "true"


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

YA_API_KEY = os.environ.get('YA_API_KEY')
YA_FOLDER_ID = os.environ.get('YA_FOLDER_ID')
YA_AUTH_TOKEN = os.environ.get('YA_AUTH_TOKEN')

GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

CHECK_RIGHTS=os.environ.get('CHECK_RIGHTS', default='False')

USERS_SHEET_ID=os.environ.get('USERS_SHEET_ID')
FEEDBACK_SHEET_ID=os.environ.get('FEEDBACK_SHEET_ID')
GOOGLE_SHEETS_CRED=os.environ.get('GOOGLE_SHEETS_CRED')


JINA_API_KEY=os.environ.get('JINA_API_KEY')


WHISPER_MODEL = os.environ.get('WHISPER_MODEL', default='small')
WHISPER_MODEL_PATH=os.environ.get('WHISPER_MODEL_PATH')

UPD_TIMEOUT = os.environ.get('UPD_TIMEOUT') or 300

def reload_admin_config():
    global CHECK_RIGHTS
    env_vars = dotenv_values(os.path.join(documents_path, 'gv.env'))
    if 'CHECK_RIGHTS' in env_vars:
        CHECK_RIGHTS=env_vars['CHECK_RIGHTS']
    else:
        CHECK_RIGHTS='False'

NOTION_INDEX_FOLDER = os.environ.get('NOTION_INDEX_FOLDER') or "./data/notion_idx"
CHATS_INDEX_FOLDER = os.environ.get('CHATS_INDEX_FOLDER') or "./data/chats_idx"
ASSISTANT_INDEX_FOLDER = os.environ.get('ASSISTANT_INDEX_FOLDER') or "./data/ass_idx"
RERANKING_MODEL = os.environ.get('RERANKING_MODEL') or '/models/bge-reranker-large'
NO_CUDA = os.environ.get('NO_CUDA', "False")

DEBUG_WORKFLOW = (os.environ.get('DEBUG_WORKFLOW', default='False').lower() == 'true')

RETRIEVER_TYPE=os.environ.get('RETRIEVER_TYPE', "teamly")
TEAM_GPT_MODEL=os.environ.get('TEAM_GPT_MODEL', "nano")

LLM_PROVIDER=os.environ.get('LLM_PROVIDER', "openai")
VALIDATION_LLM=os.environ.get('VALIDATION_LLM', "openai")
BOT_MODE=os.environ.get('BOT_MODE', "polling")
WEBHOOK_BASE=os.environ.get('WEBHOOK_BASE', "")
WEBAPP_HOST=os.environ.get('WEBAPP_HOST', "")
WEBAPP_PORT=os.environ.get('WEBAPP_PORT', "0")
WEBHOOK_PATH=os.environ.get('WEBHOOK_PATH', "")
WEBHOOK_SECRET=os.environ.get('WEBHOOK_SECRET', "")

MIN_RERANKER_RATIO=os.environ.get('MIN_RERANKER_RATIO', "0.25")

MAX_TEAMLY_DOCS=int(os.environ.get('MAX_TEAMLY_DOCS', 20))


LANGFUSE_SECRET=os.environ.get('LANGFUSE_SECRET', "")
LANGFUSE_PUBLIC=os.environ.get('LANGFUSE_PUBLIC', "")
LANGFUSE_URL=os.environ.get('LANGFUSE_URL', "")

FORCE_CPU = os.environ.get("FORCE_CPU", "1")

CLIENT_ID = os.environ.get("CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET", "")
HH_TOKEN_URL = os.environ.get("HH_TOKEN_URL", "")

def _split_env_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]

_WEBCHAT_DEFAULT_TYPES = [
    "text/plain",
    "text/markdown",
    "application/markdown",
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]

_WEBCHAT_DEFAULT_EXTENSIONS = [
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".doc",
    ".docx",
]

WEBCHAT_ALLOWED_FILE_TYPES = _split_env_list(os.environ.get("WEBCHAT_ALLOWED_FILE_TYPES")) or _WEBCHAT_DEFAULT_TYPES
WEBCHAT_ALLOWED_FILE_EXTENSIONS = _split_env_list(os.environ.get("WEBCHAT_ALLOWED_FILE_EXTENSIONS")) or _WEBCHAT_DEFAULT_EXTENSIONS
