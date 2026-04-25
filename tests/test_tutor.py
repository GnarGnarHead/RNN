from __future__ import annotations

import random
import unittest

from rnn.tutor import (
    build_lesson,
    grade_task,
    maintenance_examples,
    maintenance_targets,
    normalize_tasks,
    select_target_task,
    task_examples,
)


class TutorTaskTests(unittest.TestCase):
    def test_build_lesson_uses_contextual_next_prompts(self) -> None:
        lesson = build_lesson("next2", "A", list("ABCDEFG"))

        self.assertEqual(lesson.seq, "AB")
        self.assertEqual(lesson.prompt, "N:ABCDEFG:AB:n")
        self.assertEqual(lesson.expected, "C")

    def test_normalize_tasks_accepts_aliases_and_deduplicates(self) -> None:
        self.assertEqual(
            normalize_tasks("copy,pair,n,predict"), ["copy", "copy2", "next"]
        )

    def test_task_examples_applies_weights(self) -> None:
        examples = task_examples(
            list("AB"),
            ["copy", "next"],
            weights={"copy": 2, "next": 1},
        )

        self.assertEqual(len(examples), 6)
        self.assertEqual(examples.count({"prompt": "A", "answer": "A"}), 2)
        self.assertEqual(examples.count({"prompt": "N:AB:A:n", "answer": "B"}), 1)

    def test_maintenance_targets_include_expected_and_confused_neighborhoods(
        self,
    ) -> None:
        targets = list("ABCDEFGH")
        failures = [
            {
                "target": "G",
                "task": "next",
                "prompt": "N:ABCDEFGH:G:n",
                "expected": "H",
                "got": "A",
            }
        ]

        maint = maintenance_targets(targets, failures=failures, radius=1)

        self.assertIn("G", maint)
        self.assertIn("H", maint)
        self.assertIn("A", maint)
        self.assertIn("B", maint)

    def test_maintenance_examples_add_local_mimicry_and_task_contrasts(
        self,
    ) -> None:
        targets = list("ABCDEFGH")
        failures = [
            {
                "target": "H",
                "task": "copy",
                "prompt": "H",
                "expected": "H",
                "got": "F",
            }
        ]

        examples = maintenance_examples(
            targets,
            ["next"],
            failures=failures,
            focus_weight=1,
            radius=1,
            include_mimicry=True,
        )

        self.assertIn({"prompt": "H", "answer": "H"}, examples)
        self.assertIn({"prompt": "G", "answer": "G"}, examples)
        self.assertIn({"prompt": "N:ABCDEFGH:G:n", "answer": "H"}, examples)
        self.assertIn({"prompt": "N:ABCDEFGH:H:n", "answer": "A"}, examples)

    def test_sequential_cycle_scheduler_covers_cross_product(self) -> None:
        rng = random.Random(1337)
        seq_index = 0
        task_index = 0
        pairs = []

        for _ in range(4):
            target, task, seq_index, task_index = select_target_task(
                list("GH"),
                ["copy", "copy2"],
                order="sequential",
                task_order="cycle",
                seq_index=seq_index,
                task_index=task_index,
                rng=rng,
            )
            pairs.append((target, task))

        self.assertEqual(
            pairs,
            [
                ("G", "copy"),
                ("G", "copy2"),
                ("H", "copy"),
                ("H", "copy2"),
            ],
        )

    def test_grade_task_exact_and_kind_partial_credit(self) -> None:
        self.assertEqual(
            grade_task(
                "A",
                "A",
                task="copy",
                seq="A",
                alphabet=list("AB"),
                copy_continue_score=0.8,
                next_copy_score=0.65,
            )[0],
            1.0,
        )
        self.assertEqual(
            grade_task(
                "B",
                "A",
                task="next",
                seq="A",
                alphabet=list("AB"),
                copy_continue_score=0.8,
                next_copy_score=0.65,
            )[0],
            0.65,
        )


if __name__ == "__main__":
    unittest.main()
