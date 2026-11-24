"""Helpers for safely embedding JSON and text in generated HTML documents."""

from __future__ import annotations

import json


def json_for_script(value: object) -> str:
    """Serialize JSON that cannot terminate its containing ``<script>`` tag."""

    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
