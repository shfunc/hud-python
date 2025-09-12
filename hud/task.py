"""Task model for HUD datasets."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from string import Template
from typing import Any

from pydantic import BaseModel, Field, field_validator

from hud.settings import settings
from hud.types import MCPToolCall

logger = logging.getLogger(__name__)

