import json
import sys
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch
from io import BytesIO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from transport import remote_predict

class Response(BytesIO):
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

class TransportTests(unittest.TestCase):
    def test_api_key_is_only_sent_in_server_authorization_header(self):
        with patch('transport.urlopen',return_value=Response(b'{"cases": []}')) as http:
            self.assertEqual(remote_predict('https://demo.hf.space','secret',{'tokens':['A']}),{'cases':[]})
            request=http.call_args.args[0]
            self.assertEqual(request.full_url,'https://demo.hf.space/predict')
            self.assertEqual(request.get_header('Authorization'),'Bearer secret')
            self.assertNotIn('secret',request.data.decode())

    def test_https_required_except_loopback(self):
        with self.assertRaises(ValueError):remote_predict('http://public.example','',{})

    def test_backend_busy_returns_actionable_error(self):
        error=HTTPError('https://example/predict',429,'Busy',{},BytesIO(b'{"detail":"Model busy"}'))
        with patch('transport.urlopen',side_effect=error), self.assertRaisesRegex(ValueError,'429'):
            remote_predict('https://example','',{})
