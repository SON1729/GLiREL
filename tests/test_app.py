import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest


class HostTests(unittest.TestCase):
    def test_default_and_json_modes_render_without_model(self):
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        app.radio[0].set_value('Nhập JSON từ Kaggle').run()
        self.assertEqual(len(app.exception), 0)

    def test_live_form_does_not_load_model_until_submit(self):
        sys.modules.pop('inference', None)
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
        app.radio[0].set_value('Chạy checkpoint').run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn('inference', sys.modules)
        self.assertEqual(len(app.text_area), 4)


if __name__ == '__main__':
    unittest.main()
