from __future__ import absolute_import

import unittest

from native_anchor_gate import admit_candidate


class NativeAnchorGateTest(unittest.TestCase):
    def setUp(self):
        self.native_logits = [0.2, 1.0, 0.8]
        self.native_embeddings = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
        self.candidate_embeddings = [[0.0, 0.98], [1.0, 0.0]]

    def test_abstains_when_native_memory_changes(self):
        result = admit_candidate(
            self.native_logits, self.native_embeddings,
            [1.8, 1.7], self.candidate_embeddings,
            [[1.8, 1.7], [1.8, 1.7]], native_drift=1.0)
        self.assertEqual(result["selected_source"], "native")
        self.assertEqual(result["reason"], "no_candidate_passed_certificate")

    def test_rejects_ood_score_excursion(self):
        result = admit_candidate(
            self.native_logits, self.native_embeddings,
            [2.0, 1.1], self.candidate_embeddings,
            [[2.0, 1.1], [2.0, 1.1]], native_drift=0.0)
        self.assertEqual(result["selected_source"], "candidate")
        self.assertEqual(result["selected_index"], 4)
        self.assertIn("ood_score_excess", result["candidates"][0]["reasons"])

    def test_admits_supported_stable_candidate(self):
        result = admit_candidate(
            self.native_logits, self.native_embeddings,
            [1.2, 0.9], self.candidate_embeddings,
            [[1.18, 0.9], [1.22, 0.9]], native_drift=0.0)
        self.assertEqual(result["selected_source"], "candidate")
        self.assertEqual(result["selected_index"], 3)

    def test_abstains_on_unstable_candidate(self):
        result = admit_candidate(
            self.native_logits, self.native_embeddings,
            [1.2, 0.9], self.candidate_embeddings,
            [[0.8, 0.9], [1.6, 0.9]], native_drift=0.0)
        self.assertEqual(result["selected_source"], "native")
        self.assertIn("unstable_candidate_score", result["candidates"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
