from __future__ import annotations

import unittest

from eval_core.local.protocol import LocalRequest, encode_prompt
from eval_core.local.screening import final_answer, score_answer


class TokenizerFixture:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return self.result


class ScreeningTests(unittest.TestCase):
    def test_only_final_answer_is_scored_after_reasoning(self):
        raw = "I first considered 0.</think>1<|im_end|>"
        final = final_answer(raw, thinking_prefix=True)
        result = score_answer(dict(kind="bit", expected="1"), final)
        self.assertEqual(final, "1")
        self.assertTrue(result["exact_correct"])
        self.assertTrue(result["semantic_correct"])

    def test_unfinished_reasoning_is_not_a_final_answer(self):
        self.assertIsNone(final_answer("The answer might be 0", thinking_prefix=True))
        self.assertIsNone(final_answer("<think>The answer might be 0"))

    def test_prose_and_exact_format_are_separate(self):
        result = score_answer(dict(kind="color", expected="red"), "The color is red.")
        self.assertTrue(result["semantic_correct"])
        self.assertFalse(result["exact_correct"])

    def test_ambiguous_mapping_is_not_selected_arbitrarily(self):
        result = score_answer(dict(kind="bit", expected="1"), "red is 0 and blue is 1")
        self.assertIsNone(result["extracted"])
        self.assertFalse(result["semantic_correct"])

    def test_tokenizer_list_and_dictionary_apis_are_equivalent(self):
        request = LocalRequest("fixture", 2, "prompt", 100, 42, "native-chat-thinking-v1")
        for encoded in ([1, 2, 3], {"input_ids": [1, 2, 3]}, {"input_ids": [[1, 2, 3]]}):
            tokenizer = TokenizerFixture(encoded)
            self.assertEqual(encode_prompt(tokenizer, request), [1, 2, 3])
            self.assertTrue(tokenizer.kwargs["enable_thinking"])

    def test_batched_prompt_is_rejected(self):
        request = LocalRequest("fixture", 2, "prompt", 100, 42, "native-chat-system-v1")
        with self.assertRaisesRegex(ValueError, "batch size one"):
            encode_prompt(TokenizerFixture({"input_ids": [[1], [2]]}), request)


if __name__ == "__main__":
    unittest.main()
