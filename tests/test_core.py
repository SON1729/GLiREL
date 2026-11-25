import copy
import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import validate_bundle,validate_example,select,evaluate,diagnosis,highlighted

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.bundle=validate_bundle(json.loads((ROOT/'data/recorded.json').read_text(encoding='utf-8')))
        self.case=self.bundle['cases'][0]

    def test_all_recorded_examples_are_consistent(self):
        self.assertEqual(len(self.bundle['cases']),5)
        for case in self.bundle['cases']:
            ex=case['example']
            self.assertEqual(len(case['candidates_before_display_thresholds']),
                len(ex['entities'])*(len(ex['entities'])-1)*len(ex['labels']))

    def test_top_k_is_per_directed_pair_not_global(self):
        c=self.case['candidates_before_display_thresholds']
        self.assertEqual(len(select(c,0,1)),2)
        self.assertEqual(len(select(c,0,-1)),6)

    def test_strict_threshold_and_no_softmax(self):
        p=copy.deepcopy(self.case['candidates_before_display_thresholds'][0]);p['score']=.5
        self.assertEqual(select([p],.5),[])
        self.assertEqual(len(select([p],.49)),1)

    def test_direction_and_span_end_conventions(self):
        ex=self.case['example']
        expected=[p for p in self.case['candidates_before_display_thresholds']
                  if p['label']=='employer' and p['head_pos']==[0,1] and p['tail_pos']==[3,4]]
        self.assertEqual(len(expected),1)
        self.assertEqual(evaluate(ex,expected)['tp'],1)
        reverse=copy.deepcopy(expected[0]);reverse['head_pos'],reverse['tail_pos']=reverse['tail_pos'],reverse['head_pos']
        self.assertEqual(evaluate(ex,[reverse])['tp'],0)

    def test_missing_label_is_structural_not_threshold(self):
        case=self.bundle['cases'][3];ex=copy.deepcopy(case['example'])
        ex['expected']=[[0,'employer',1]]
        notes=diagnosis(ex,case['candidates_before_display_thresholds'],[],.3)
        self.assertTrue(any('không có trong danh sách' in n[2] for n in notes))

    def test_invalid_spans_rejected_before_model(self):
        ex=copy.deepcopy(self.case['example']);ex['entities'][0][1]=999
        with self.assertRaises(ValueError):validate_example(ex)

    def test_html_input_is_escaped(self):
        ex=copy.deepcopy(self.case['example']);ex['tokens'][0]='<script>alert(1)</script>'
        self.assertNotIn('<script>',highlighted(ex))

    def test_import_nonfinite_score_rejected(self):
        b=copy.deepcopy(self.bundle);b['cases'][0]['candidates_before_display_thresholds'][0]['score']=float('nan')
        with self.assertRaises(ValueError):validate_bundle(b)

if __name__=='__main__':unittest.main()
