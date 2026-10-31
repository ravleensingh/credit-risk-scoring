from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent.nodes import DEFAULT_GROQ_MODEL, _get_llm


class GroqConfigurationTests(unittest.TestCase):
    def test_default_model_uses_current_groq_replacement(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True):
            with patch("agent.nodes.ChatGroq") as chat_groq:
                _get_llm()

        chat_groq.assert_called_once_with(
            model=DEFAULT_GROQ_MODEL,
            temperature=0.2,
            api_key="test-key",
        )

    def test_model_can_be_overridden_by_environment(self):
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test-key", "GROQ_MODEL": "account-specific-model"},
            clear=True,
        ):
            with patch("agent.nodes.ChatGroq") as chat_groq:
                _get_llm()

        self.assertEqual(chat_groq.call_args.kwargs["model"], "account-specific-model")

    def test_missing_api_key_is_reported_without_request(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY is not configured"):
                _get_llm()
