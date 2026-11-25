import copy
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
try:
    from fastapi.testclient import TestClient
    import service
except ImportError:
    service=None
from core import validate_bundle

@unittest.skipIf(service is None,'Optional backend test dependencies are not installed')
class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(service.app)
        self.record=validate_bundle(json.loads((ROOT/'data/recorded.json').read_text(encoding='utf-8')))['cases'][0]

    def test_health_does_not_load_model(self):
        with patch('service.get_model') as model:
            model.cache_info.return_value=types.SimpleNamespace(currsize=0)
            self.assertEqual(self.client.get('/health').json()['model_loaded'],False)
            model.assert_not_called()

    def test_unauthorized_and_busy_requests_do_not_infer(self):
        with patch.dict('os.environ',{'GLIREL_API_KEY':'example-key'}):
            self.assertEqual(self.client.post('/predict',json={'example':self.record['example']}).status_code,401)
            service.BUSY.acquire()
            try:
                r=self.client.post('/predict',json={'example':self.record['example']},headers={'Authorization':'Bearer example-key'})
                self.assertEqual(r.status_code,429)
            finally:service.BUSY.release()

    def test_contract_returns_same_input_with_trace(self):
        # Contract test only: inference is replaced by an explicit recorded fixture.
        fake=types.ModuleType('inference')
        fake.model_config=lambda model: {'scorer':'dot'}
        fake.predict=lambda model,example: {**copy.deepcopy(self.record),'example':example}
        with patch.dict('os.environ',{'GLIREL_API_KEY':''}),patch.dict(sys.modules,{'inference':fake}),patch('service.get_model',return_value=object()):
            r=self.client.post('/predict',json={'example':self.record['example']})
        self.assertEqual(r.status_code,200)
        result=validate_bundle(r.json())
        self.assertEqual(result['cases'][0]['example'],self.record['example'])

    def test_invalid_span_rejected_before_model_loading(self):
        ex=copy.deepcopy(self.record['example']);ex['entities'][0][1]=999
        with patch.dict('os.environ',{'GLIREL_API_KEY':''}),patch('service.get_model') as model:
            self.assertEqual(self.client.post('/predict',json={'example':ex}).status_code,422)
            model.assert_not_called()
