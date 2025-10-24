from typing import Any
#import torch
#import torchaudio
import whisper
import os
import torch

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

#from pyannote.audio import Pipeline
#import os

import logging
logger = logging.getLogger(__name__)

def pick_device() -> str:
    # Allow forcing CPU from env if needed: FORCE_CPU=1 python app.py
    if config.FORCE_CPU == "1":
        return "cpu"

    # Prefer CUDA only if the GPU arch is new enough (cc >= 7.0)
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 7:
                return "cuda"
            else:
                logger.warning(f"GPU compute capability {major}.{minor} detected; "
                               "using CPU to avoid CUDA arch mismatch.")
        except Exception as e:
            logger.warning(f"Could not query CUDA capability: {e}")

    # macOS Metal fallback
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"

DEVICE = pick_device()
FP16 = (DEVICE == "cuda")  # never use fp16 on cpu/mps


WHISPER_MODEL=config.WHISPER_MODEL
#WHISPER_MODEL="small"
WHISPER_MODEL_PATH = config.WHISPER_MODEL_PATH
logger.info(f"Loading whisper model: {WHISPER_MODEL}")
logger.info(f"Loading whisper model '{WHISPER_MODEL}' on device: {DEVICE}")
try:
    model = whisper.load_model(WHISPER_MODEL,
                               download_root=WHISPER_MODEL_PATH,
                               device=DEVICE)
except Exception as e:
    # Catch CUDA arch errors and fall back to CPU automatically
    logger.warning(f"Whisper load failed on {DEVICE}: {e}. Falling back to CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide GPU for this process
    DEVICE, FP16 = "cpu", False
    model = whisper.load_model(WHISPER_MODEL,
                               download_root=WHISPER_MODEL_PATH,
                               device="cpu")
logger.info(f"Whisper model {WHISPER_MODEL} loaded")



def recognise_text(audio_path: Any) -> str:
    script = model.transcribe(audio_path)
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if DEVICE == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return script["text"] if "text" in script else ""

if __name__ == '__main__':
    print(recognise_text("voices/audio_2024-11-06_18-04-50.ogg"))
