"""Offline regressions for the reported delivery patches; no network clients."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidetrace_memory import PptMemory


def embed(_text):
    return [1.0, 0.25, 0.125, 0.5]


class ForgetRegression(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.memory = PptMemory(Path(self.directory.name) / "memory.db",
                                user_id="regression-user", embedder=embed)
        self.memory.start_project("p1", "人物介绍", page_ids=["one", "two"])

    def tearDown(self):
        self.directory.cleanup()

    def add(self, **extras):
        return self.memory.feedback_claim({
            "subject": "人物头像", "attribute": "形状",
            "action_or_value": "圆形", "direction": "prefer", **extras,
        }, round_no=1)

    def test_forget_unpromoted_working_item(self):
        wid = self.add()["working_claim_id"]
        self.assertIn(wid, [c["claim_id"] for c in self.memory.context(round_no=2)["claims"]])
        self.assertTrue(self.memory.forget(wid)["changed"])
        self.assertNotIn(wid, [c["claim_id"] for c in self.memory.context(round_no=3)["claims"]])

    def test_forget_unpromoted_revision_chain(self):
        wid = self.add()["working_claim_id"]
        revised = self.memory.revise(wid, round_no=2, action_or_value="方形")["working_claim_id"]
        self.memory.context(round_no=3)
        self.memory.forget(wid)
        self.assertEqual([], self.memory.context(round_no=4)["claims"])
        states = {c["working_claim_id"]: c["status"] for c in
                  self.memory.list_memories(include_working=True)["working"]}
        self.assertEqual("retracted", states[wid])
        self.assertEqual("retracted", states[revised])

    def test_forget_canonical_after_revision(self):
        wid = self.add()["working_claim_id"]
        self.memory.context(round_no=2)
        self.memory.record_outcome(round_no=2, implemented_ids=[wid], accepted_ids=[wid])
        self.memory.finish_project(satisfied=True, final_round=2)
        cid = self.memory.list_memories()["canonical"][0]["claim_id"]
        self.memory.start_project("p2", "新人物介绍")
        self.memory.revise(cid, round_no=1, action_or_value="方形")
        self.memory.context(round_no=2)
        self.memory.forget(cid)
        self.assertEqual([], self.memory.context(round_no=3)["claims"])

    def test_structured_analyzer_fields(self):
        result = self.add(partition="drawing", partition_reason="头像视觉",
                          classification="stable_preference", priority="should",
                          confidence=0.9, priority_update=False,
                          relation={"type": "unrelated", "candidate_id": "",
                                    "confidence": 0.0, "explicit": False})
        self.assertTrue(result["stored"])

    def test_explicit_priority_flag_overrides_inline(self):
        wid = self.add(priority="should")["working_claim_id"]
        self.memory.feedback_claim({
            "subject": "人物头像", "attribute": "形状", "action_or_value": "圆形",
            "direction": "prefer", "priority": "must", "priority_update": True,
            "relation": {"candidate_id": wid, "type": "reinforcement",
                         "confidence": 1.0, "explicit": True},
        }, round_no=2, priority_update=False)
        item = next(c for c in self.memory.context(round_no=3)["claims"] if c["claim_id"] == wid)
        self.assertEqual("should", item["priority"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
