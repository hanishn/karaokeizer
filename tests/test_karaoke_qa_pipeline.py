import unittest

from karaokeizer.build_karaoke_escalation_ledger import merged_regions
from karaokeizer.remediate_karaoke_regions import replacement_segments, validate_replacement


class KaraokeQaPipelineTests(unittest.TestCase):
    def test_punctuation_only_whisper_artifacts_are_discarded(self):
        result = {"segments": [{"start": 0, "end": 1, "text": ".", "words": [{"word": ".", "start": 0.5, "end": 0.5}]}]}
        self.assertEqual(replacement_segments(result, 10), [])

    def test_collapsed_replacement_is_rejected_before_merge(self):
        segments = [{"start": 10, "end": 11, "words": [{"text": "word", "start": 10.5, "end": 10.5}]}]
        self.assertIn("segment_0/word_0:collapsed", validate_replacement(segments, 10, 11))

    def test_clean_replacement_passes_transaction_gate(self):
        segments = [{"start": 10, "end": 11, "words": [{"text": "word", "start": 10.2, "end": 10.8}]}]
        self.assertEqual(validate_replacement(segments, 10, 11), [])

    def test_failure_windows_are_padded_and_merged(self):
        failures = [{"start": 10, "end": 10}, {"start": 11, "end": 11.2}, {"start": 30, "end": 30}]
        regions = merged_regions(failures, padding=1.5, gap=1.0)
        self.assertEqual(len(regions), 2)
        self.assertEqual((regions[0]["start"], regions[0]["end"]), (8.5, 12.7))


if __name__ == "__main__":
    unittest.main()
