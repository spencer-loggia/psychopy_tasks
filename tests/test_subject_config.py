import unittest

from bin.config import resolve_subject_mapped_value


class SubjectMappedConfigTests(unittest.TestCase):
    def test_resolves_exact_subject_key(self):
        value = {
            "Yuri": "freq_space_TY.csv",
            "Buzz": "freq_space_SB.csv",
        }

        self.assertEqual(
            resolve_subject_mapped_value(
                value,
                subject="Yuri",
                field_name="freq_space_tsv",
            ),
            "freq_space_TY.csv",
        )

    def test_rejects_subject_missing_from_map(self):
        with self.assertRaisesRegex(
            ValueError,
            "no entry for subject 'Sally'.*Available subjects: Buzz, Yuri",
        ):
            resolve_subject_mapped_value(
                {"Yuri": "TY.csv", "Buzz": "SB.csv"},
                subject="Sally",
                field_name="reward_space_tsv",
            )

    def test_rejects_unset_subject(self):
        with self.assertRaisesRegex(ValueError, "config field 'subject' is not set"):
            resolve_subject_mapped_value(
                {"Yuri": "TY.csv"},
                subject=None,
                field_name="freq_space_tsv",
            )

    def test_rejects_legacy_scalar_path(self):
        with self.assertRaisesRegex(ValueError, "must be a subject-to-path object"):
            resolve_subject_mapped_value(
                "freq_space_TY.csv",
                subject="Yuri",
                field_name="freq_space_tsv",
            )

    def test_rejects_empty_selected_path(self):
        with self.assertRaisesRegex(ValueError, "must be a path string"):
            resolve_subject_mapped_value(
                {"Yuri": ""},
                subject="Yuri",
                field_name="freq_space_tsv",
            )


if __name__ == "__main__":
    unittest.main()
