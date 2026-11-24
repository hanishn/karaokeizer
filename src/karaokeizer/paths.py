"""Cross-platform validation for user-derived output filenames."""

from __future__ import annotations

import re

_UNSAFE_COMPONENT = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_component(value: object, *, field: str) -> str:
    """Return a portable filename component or raise a descriptive error."""

    text = str(value)
    if not text or text in {".", ".."} or text != text.rstrip(" .") or _UNSAFE_COMPONENT.search(text):
        raise ValueError(f"{field} is not a safe cross-platform filename component: {text!r}")
    return text
