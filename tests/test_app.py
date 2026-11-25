import sys
import json
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

    def test_web_ignores_old_remote_settings_and_does_not_load_torch(self):
        with patch.dict('os.environ',{'GLIREL_API_URL':'https://example.org','GLIREL_LOCAL_INFERENCE':'0'}):
            at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertEqual(len(at.exception),0)
        self.assertNotIn('torch',sys.modules)
        args=json.loads(at.get('component_instance')[0].proto.json_args)
        self.assertEqual(args['backend'],'recorded')

    def test_local_mode_requires_explicit_flag_but_not_eager_model_load(self):
        with patch.dict('os.environ',{'GLIREL_LOCAL_INFERENCE':'1'}):
            at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertEqual(len(at.exception),0)
        args=json.loads(at.get('component_instance')[0].proto.json_args)
        self.assertIn(args['backend'],('local','local_setup'))
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
