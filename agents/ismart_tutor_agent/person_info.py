from __future__ import annotations

import logging
from typing import Optional

from .state import PersonProfile

LOG = logging.getLogger(__name__)


def get_person_info(person_id: str) -> Optional[PersonProfile]:
    """
    Retrieve a person's profile by ID.

    NOTE: This is a placeholder implementation. Integrate with your real data source
    (DB/service) and return a dict with keys: name, age, school_year, nosology_type.
    """
    LOG.info("get_person_info called for person_id=%s (no data source configured)", person_id)
    return None

