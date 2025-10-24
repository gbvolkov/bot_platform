from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db import get_session


@dataclass
class UserContext:
    user_id: str
    user_role: str


async def user_context_dependency(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> UserContext:
    user_id = x_user_id or "anonymous"
    role = x_user_role or settings.default_user_role
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing user identity.")
    return UserContext(user_id=user_id, user_role=role)


DbSession = Annotated[AsyncSession, Depends(get_session)]
UserContextDep = Annotated[UserContext, Depends(user_context_dependency)]

