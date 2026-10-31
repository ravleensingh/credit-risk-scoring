from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import rag
from utils.preprocessing import load_data


class ProjectPathTests(unittest.TestCase):
    def tearDown(self):
        rag.reset_retriever_cache()

    def test_dataset_load_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as working_directory:
            previous_directory = Path.cwd()
            try:
                os.chdir(working_directory)
                data = load_data()
            finally:
                os.chdir(previous_directory)

        self.assertEqual(len(data), 20_000)

    def test_regulation_retrieval_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as working_directory:
            previous_directory = Path.cwd()
            try:
                os.chdir(working_directory)
                with patch(
                    "agent.rag._try_build_semantic_retriever",
                    side_effect=RuntimeError("Use the local TF-IDF fallback in this test."),
                ):
                    result = rag.retrieve("credit score lending guidelines", k=2)
            finally:
                os.chdir(previous_directory)

        self.assertTrue(result)
        self.assertIn("credit", result.lower())
