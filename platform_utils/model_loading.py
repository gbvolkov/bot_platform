"""Coordinate neural-model constructors that use process-global torch contexts."""
from threading import RLock

# Transformers/accelerate change global tensor-construction state while loading
# weights. Serialize construction, without serializing inference or sharing
# privacy sessions between agents. Callers run heavy work on runtime threads.
MODEL_LOADING_LOCK = RLock()
