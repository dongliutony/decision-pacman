"""Standard-library tests for build_sft.py: cd training && python3 -m unittest test_build_sft"""
import json
import os
import tempfile
import unittest

from build_sft import encoding, label_target, load_records


def record(seed, state, label, values=None, alt_state=None):
    r = {"seed": seed, "tick": 1, "player": "oracle-5s", "encoder": "features", "state": state,
         "instructions": "rules", "criteria": {"up": None, "back": "turn around now"}, "label": label, "values": values}
    if alt_state is not None:
        r["alt"] = {"features-peek5s": {"state": alt_state, "instructions": "rules + lookahead",
                                        "criteria": {"up": None, "back": "turn around now"}}}
    return r


class LabelTargetTest(unittest.TestCase):
    def test_smoothing_spreads_over_other_options(self):
        self.assertEqual(label_target("b", ["a", "b", "c"], 0.1), [0.05, 0.9, 0.05])

    def test_no_smoothing_is_one_hot(self):
        self.assertEqual(label_target("a", ["a", "b"], 0.0), [1.0, 0.0])


class LoadRecordsTest(unittest.TestCase):
    def write(self, rows):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "run"))
        with open(os.path.join(d, "run", "states.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return d

    def test_hard_labels_and_alt_encoding(self):
        d = self.write([
            record(1, {"x": 1}, "up", alt_state={"x": 1, "peek": 1}),
            record(1, {"x": 1}, "up", alt_state={"x": 1, "peek": 2}),  # same features, different lookahead
            record(1, {"x": 2}, "back", alt_state={"x": 2, "peek": 1}),
        ])
        self.assertEqual(len(load_records(d, "label")), 2)
        rows = load_records(d, "label", "features-peek5s", smoothing=0.1)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["instructions"], "rules + lookahead")
        self.assertEqual(rows[2]["target"], [0.1, 0.9])

    def test_values_skip_records_without_values(self):
        d = self.write([record(1, {"x": 1}, "up"), record(1, {"x": 2}, "up", values={"up": 10, "back": 0})])
        rows = load_records(d, "values", temperature=50)
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["target"][0], rows[0]["target"][1])

    def test_missing_alt_encoding_is_an_error(self):
        with self.assertRaises(KeyError):
            encoding(record(1, {"x": 1}, "up"), "features-peek5s")


if __name__ == "__main__":
    unittest.main()


class RunOrderTest(unittest.TestCase):
    def write_run(self, root, name, rows):
        os.makedirs(os.path.join(root, name))
        with open(os.path.join(root, name, "states.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_duplicates_resolve_the_same_whatever_the_start_times(self):
        # The same state reached by two players; which copy is kept must not depend on
        # the timestamps that start each run folder's name.
        kept = []
        for oracle_stamp, random_stamp in [("2026-01-01T00-00-00-001Z", "2026-01-01T00-00-00-002Z"),
                                           ("2026-01-01T00-00-00-002Z", "2026-01-01T00-00-00-001Z")]:
            d = tempfile.mkdtemp()
            self.write_run(d, f"{oracle_stamp}_oracle-pellets-5s_oracle-pellets-5s_features_s900_p11",
                           [record(900, {"x": 1}, "up", values={"up": 2, "back": 1})])
            self.write_run(d, f"{random_stamp}_random_oracle-pellets-5s_features_s940_p12",
                           [record(940, {"x": 1}, "back", values={"up": 1, "back": 2})])
            kept.append([r["seed"] for r in load_records(d, "values")])
        self.assertEqual(kept[0], kept[1])
        self.assertEqual(len(kept[0]), 1)
