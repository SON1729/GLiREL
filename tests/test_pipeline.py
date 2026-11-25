import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit_pipeline import audit_bundle

class PipelineAuditTests(unittest.TestCase):
    def setUp(self):
        self.bundle=json.loads((ROOT/'data/recorded.json').read_text(encoding='utf-8'))

    def test_real_saved_logits_and_shapes(self):
        report=audit_bundle(self.bundle)
        self.assertEqual(report['checked_logits'],28)
        self.assertEqual(len(report['cases']),5)
        self.assertLess(max(c['max_sigmoid_error'] for c in report['cases']),4e-8)
        self.assertIn('span_marker_mode',report['missing_configuration'])

    def test_wrong_label_axis_cannot_pass(self):
        bad=copy.deepcopy(self.bundle)
        bad['cases'][0]['example']['labels'].reverse()
        with self.assertRaises(ValueError):audit_bundle(bad)

    def test_wrong_score_cannot_pass(self):
        bad=copy.deepcopy(self.bundle)
        bad['cases'][0]['candidates_before_display_thresholds'][0]['score']=.2
        with self.assertRaises(ValueError):audit_bundle(bad)

    def test_wrong_pooling_dimensions_cannot_pass(self):
        bad=copy.deepcopy(self.bundle)
        bad['cases'][0]['trace']['pooled_embeddings']['shape'][2]=1024
        with self.assertRaises(ValueError):audit_bundle(bad)
