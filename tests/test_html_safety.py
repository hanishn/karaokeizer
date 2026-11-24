from karaokeizer.html_safety import json_for_script


def test_script_json_escapes_html_terminators_and_ampersands():
    payload = json_for_script({"title": "</script><img src=x>&"})
    assert "</script>" not in payload
    assert "<img" not in payload
    assert "&" not in payload
    assert "\\u003c/script\\u003e" in payload


def test_script_json_escapes_javascript_line_separators():
    payload = json_for_script({"text": "before\u2028middle\u2029after"})
    assert "\u2028" not in payload
    assert "\u2029" not in payload
    assert "\\u2028" in payload
    assert "\\u2029" in payload
