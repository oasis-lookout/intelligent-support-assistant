import unittest

from support_assistant.config import Settings


class ConfigTests(unittest.TestCase):
    def test_mode_defaults(self):
        s = Settings.from_env({"DEPLOYMENT_MODE": "local"})
        self.assertEqual((s.embedding_provider, s.llm_provider, s.vector_store), ("hash", "extractive", "memory"))
        a = Settings.from_env({"DEPLOYMENT_MODE": "aws", "BEDROCK_LLM_MODEL_ID": "some.model-id"})
        self.assertEqual((a.embedding_provider, a.vector_store, a.embedding_dim), ("bedrock", "qdrant", 1024))

    def test_validation(self):
        with self.assertRaises(ValueError):
            Settings.from_env({"DEPLOYMENT_MODE": "aws"})                     # model id missing
        with self.assertRaises(ValueError):
            Settings.from_env({"APP_ENV": "prod"})                            # dev secret outside dev
        with self.assertRaises(ValueError):
            Settings.from_env({"DEPLOYMENT_MODE": "selfhosted"})              # LLM_MODEL missing
        with self.assertRaises(ValueError):
            Settings.from_env({"DEPLOYMENT_MODE": "nope"})
        Settings.from_env({"APP_ENV": "prod", "SESSION_SECRET": "s" * 40})


if __name__ == "__main__":
    unittest.main()
