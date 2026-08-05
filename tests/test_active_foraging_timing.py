import unittest

from task.active_foraging_timing import (
    duration_requires_positive_frames,
    validate_duration_for_presentation_mode,
)


class ActiveForagingTimingTests(unittest.TestCase):
    def test_duration_must_be_positive_for_memory_even_when_simultaneous(self):
        self.assertTrue(duration_requires_positive_frames(sequential=False, is_memory=True))
        validate_duration_for_presentation_mode(0.05, sequential=False, is_memory=True)

        with self.assertRaises(ValueError):
            validate_duration_for_presentation_mode(0.0, sequential=False, is_memory=True)

    def test_duration_stays_zero_for_simultaneous_non_memory(self):
        self.assertFalse(duration_requires_positive_frames(sequential=False, is_memory=False))
        validate_duration_for_presentation_mode(0.0, sequential=False, is_memory=False)

        with self.assertRaises(ValueError):
            validate_duration_for_presentation_mode(0.05, sequential=False, is_memory=False)

    def test_duration_must_be_positive_for_sequential(self):
        self.assertTrue(duration_requires_positive_frames(sequential=True, is_memory=False))
        self.assertTrue(duration_requires_positive_frames(sequential=True, is_memory=True))

        with self.assertRaises(ValueError):
            validate_duration_for_presentation_mode(0.0, sequential=True, is_memory=False)


if __name__ == "__main__":
    unittest.main()
