import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.utils import load_models, predict_message


class TestSpamModelCompatibility(unittest.TestCase):
    def test_predict_message_works_with_serialized_svc(self):
        model, vectorizer, _ = load_models()
        self.assertIsNotNone(model)
        self.assertIsNotNone(vectorizer)

        prediction, probability = predict_message(
            'Congratulations! You have won a free prize.',
            model,
            vectorizer,
        )

        self.assertIn(prediction, (0, 1))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)


if __name__ == '__main__':
    unittest.main()
