import json

from karaokeizer.audit_karaoke_transcripts import audit_transcript
from karaokeizer.compare_karaoke_transcripts import edit_distance, normalized_words


def write_transcript(path, segments):
    path.write_text(json.dumps({"track": 1, "title": "Fixture", "segments": segments}), encoding="utf-8")


def test_audit_reports_duplicate_and_collapsed_timing(tmp_path):
    path = tmp_path / "01 - Fixture.whisper.json"
    write_transcript(
        path,
        [{
            "start": 0.0,
            "end": 1.0,
            "words": [
                {"text": "one", "start": 0.2, "end": 0.2},
                {"text": "two", "start": 0.2, "end": 0.8},
            ],
        }],
    )
    result = audit_transcript(path)
    kinds = {failure["type"] for failure in result["failures"]}
    assert {"collapsed_word", "duplicate_word_start"} <= kinds
    assert not result["pass"]


def test_normalized_words_are_case_and_punctuation_insensitive():
    data = {"segments": [{"words": [{"text": "Hello!", "start": 0, "end": 1}, {"text": "WORLD", "start": 1, "end": 2}]}]}
    assert normalized_words(data) == ["hello", "world"]


def test_edit_distance_has_expected_insert_delete_substitute_costs():
    assert edit_distance(["a", "b"], ["a", "b"]) == 0
    assert edit_distance(["a", "b"], ["a", "c"]) == 1
    assert edit_distance(["a"], ["a", "b"]) == 1
