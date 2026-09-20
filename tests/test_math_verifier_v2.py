import unittest

from final_execution.math_verifier import extract_candidates, verify_answers


class MathVerifierV2Tests(unittest.TestCase):
    def test_final_box_wins(self):
        result = verify_answers("36", r"Intermediate \boxed{25}; final \boxed{36}")
        self.assertTrue(result["correct"])
        self.assertEqual(result["extracted_prediction"], ["36"])

    def test_thousands_separator_is_scalar(self):
        self.assertTrue(verify_answers("58,500", r"Final: \boxed{58500}")["correct"])

    def test_decimal_fraction_equivalence(self):
        self.assertTrue(verify_answers(r"\frac{14}{5}", r"\boxed{2.8}")["correct"])

    def test_percent_display_value(self):
        self.assertTrue(verify_answers("10", r"\boxed{10\%}")["correct"])

    def test_simple_assignment(self):
        self.assertTrue(verify_answers(r"\frac{8}{5}", r"\boxed{a=\frac{8}{5}}")["correct"])

    def test_missing_set_member_rejected(self):
        self.assertFalse(verify_answers("-1,0,1", r"\boxed{0,1}")["correct"])

    def test_wrong_tuple_rejected(self):
        self.assertFalse(verify_answers("-3,0", r"\boxed{0 \text{ and } 0}")["correct"])

    def test_interval_subset_rejected(self):
        self.assertFalse(verify_answers(r"(0,9) \cup (9,36)", r"\boxed{(9,36)}")["correct"])

    def test_open_closed_interval_difference_rejected(self):
        self.assertFalse(verify_answers(r"(-\infty,0]", r"\boxed{(-\infty,0)}")["correct"])

    def test_membership_prefix_is_ignored(self):
        self.assertTrue(verify_answers(r"x \in [-2,7]", r"\boxed{[-2,7]}")["correct"])

    def test_structured_prose_answer(self):
        text = "Final Answer: Thurka bought 7 stuffed goats and 4 toy helicopters."
        self.assertTrue(verify_answers("7,4", text)["correct"])

    def test_multiple_equality_not_scalar(self):
        self.assertFalse(verify_answers("60", r"\boxed{\angle B=\angle E=60^\circ}")["correct"])

    def test_rationale_number_without_terminal_answer_rejected(self):
        text = r"We know $\angle B=60^\circ$. We still need to construct the requested triangle."
        self.assertFalse(verify_answers("60", text)["correct"])
        self.assertEqual(extract_candidates(text).failure_type, "no_candidate")

    def test_terminal_conclusion_is_accepted(self):
        text = "Work. Therefore, the number of representatives is 10."
        self.assertTrue(verify_answers("10", text)["correct"])

    def test_terminal_structured_conclusion_is_accepted(self):
        text = "Work. Therefore, the possible values are 1, 3, 5, and 15."
        self.assertTrue(verify_answers("1,3,5,15", text)["correct"])

    def test_terminal_conclusion_without_copula_is_accepted(self):
        text = "Work. Therefore, Jim walks 200 feet less than Martha."
        self.assertTrue(verify_answers("200", text)["correct"])

    def test_terminal_fraction_without_copula_is_accepted(self):
        text = r"Work. Therefore, they eat $\frac{13}{15}$ of the pie altogether."
        self.assertTrue(verify_answers(r"\frac{13}{15}", text)["correct"])

    def test_currency_answer_is_accepted(self):
        text = r"Work. Therefore, the original price was $\$36$."
        self.assertTrue(verify_answers(r"\$36", text)["correct"])

    def test_multiline_display_math_after_final_answer(self):
        text = "### Final Answer:\n$$\n\\sqrt{53}\n$$\nThis is already in simplest form."
        self.assertTrue(verify_answers(r"\sqrt{53}", text)["correct"])

    def test_exponent_is_not_truncated_by_symbolic_parser(self):
        self.assertFalse(verify_answers("2", r"\boxed{2^{100}}")['correct'])


if __name__ == "__main__":
    unittest.main()
