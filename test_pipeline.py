# test_pipeline.py
import unittest
import os
from dotenv import load_dotenv
from preprocessor import clean_text

# Load variables safely for automated evaluation runs
load_dotenv(override=True)

class TestRAGEngineComponents(unittest.TestCase):
    
    def test_regex_preprocessor_cleanup(self):
        """Verify structural noise patterns are successfully removed from documents."""
        dirty_input = "Confidential Page 12 of 45 Draft - Attention mechanisms are efficient."
        expected_output = "- Attention mechanisms are efficient."
        self.assertEqual(clean_text(dirty_input), expected_output)

    def test_environment_keys_exist(self):
        """Ensure core configuration keys are defined before test suite runtime execution."""
        self.assertIsNotNone(os.getenv("OPENAI_API_KEY"), "Missing critical OpenAI API configurations.")

if __name__ == "__main__":
    unittest.main()
