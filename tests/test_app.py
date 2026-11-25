import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest

class HostTests(unittest.TestCase):
    def test_component_renders_without_model_or_api(self):
        at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertEqual(len(at.exception),0)
        self.assertTrue(len(at.get('component_instance')))
        self.assertEqual(len(at.text_area),0)  # Editor is in the component.

    def test_remote_does_not_load_torch(self):
        with patch.dict('os.environ',{'GLIREL_API_URL':'https://example.org'}):
            at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertEqual(len(at.exception),0)
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
