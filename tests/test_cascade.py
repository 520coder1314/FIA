import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from fia.__main__ import select, digest


class CascadeTests(unittest.TestCase):
    def test_coverage_and_vlm_reranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            c, split, scores = [p/n for n in ['c.json','selection.json','scores.csv']]
            c.write_text(json.dumps({'pairs':[{'source':0,'target':i,'eligible':True} for i in [1,2]]}))
            split.write_text(json.dumps({'role':'selection','images':[]}))
            pairs=[(0,1),(1,0),(0,2),(2,0)]
            def save(stage, pairs, values):
                with scores.open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['stage','source','target','seed','split','split_sha256','asr','ftr'])
                    for (a,b),v in zip(pairs,values):
                        for seed in [42,7]:w.writerow([stage,a,b,seed,'selection',digest(split),v,0])
            args=SimpleNamespace(candidates=c,selection_split=split,scores=scores,output=p,stage='small',top_k=2,shortlist=None,proxy_model='ResNet34')
            save('small',pairs[:3],[90,70,60])
            with self.assertRaisesRegex(ValueError,'Incomplete'):select(args)
            save('small',pairs,[90,70,60,50]); select(args)
            shortlist=p/'shortlist.json'
            self.assertEqual([(v['source'],v['target']) for v in json.loads(shortlist.read_text())['shortlist']],pairs[:2])
            args.stage='vlm';args.shortlist=shortlist;args.proxy_model='Janus-Pro-7B'
            save('vlm',pairs[:1],[80])
            with self.assertRaisesRegex(ValueError,'Incomplete'):select(args)
            save('small',pairs[:2],[80,95])
            with self.assertRaisesRegex(ValueError,'stage mismatch'):select(args)
            save('vlm',pairs[:2],[80,95]);select(args)
            self.assertEqual(json.loads((p/'selected.json').read_text())['selected']['source'],1)
            split.write_text(json.dumps({'role':'selection','images':[{}]}))
            with self.assertRaisesRegex(ValueError,'provenance'):select(args)

    def test_invalid_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); c=p/'c';m=p/'m'
            c.write_text(json.dumps({'pairs':[{'source':0,'target':1,'eligible':True}]}));m.write_text('{"role":"selection"}')
            args=SimpleNamespace(candidates=c,selection_split=m,stage='small',shortlist=None,top_k=1)
            with self.assertRaisesRegex(ValueError,'top-k'):select(args)
