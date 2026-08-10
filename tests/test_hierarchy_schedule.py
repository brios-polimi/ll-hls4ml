import unittest

from ll_hls4ml.data.hierarchy import function_schedule


class FunctionScheduleTests(unittest.TestCase):
    def test_schedule_is_strictly_leaf_first(self):
        # 0 calls both 1 and 2; 1 calls 3. Functions 2 and 3 are leaves.
        schedule = function_schedule(4, [(0, 1), (0, 2), (1, 3)])

        self.assertEqual(schedule.depth, (2, 1, 0, 0))
        self.assertEqual(schedule.roots, (True, False, False, False))
        self.assertEqual(schedule.entry, (True, False, False, False))
        self.assertEqual(schedule.reachable, (True, True, True, True))

    def test_largest_reachable_root_is_the_entry_candidate(self):
        schedule = function_schedule(
            4,
            [(0, 1), (2, 3)],
            instruction_counts=[2, 20, 3, 4],
        )

        self.assertEqual(schedule.roots, (True, False, True, False))
        self.assertEqual(schedule.entry, (True, False, False, False))
        self.assertEqual(schedule.reachable, (True, True, False, False))

    def test_recursion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Recursive"):
            function_schedule(2, [(0, 1), (1, 0)])


if __name__ == "__main__":
    unittest.main()
