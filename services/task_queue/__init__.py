from __future__ import annotations

__all__ = ["RedisTaskQueue"]

def __getattr__(name):
    if name == "RedisTaskQueue":
        from .redis_queue import RedisTaskQueue
        return RedisTaskQueue
    raise AttributeError(name)

