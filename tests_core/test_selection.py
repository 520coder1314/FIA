import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from fia_core.__main__ import select, split, digest

class SelectionTests(unittest.TestCase):
    def test_empirical_margin_not_asr_or_distance(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c=p/'c.json';s=p/'selection.json';f=p/'scores.csv'
            c.write_text(json.dumps({'pairs':[{'source':0,'target':1,'eligible':True}]}))
            s.write_text(json.dumps({'role':'selection','images':[]}))
            rows=[['source','target','seed','split','split_sha256','asr','ftr'],
                  [0,1,42,'selection',digest(s),95,50],[1,0,42,'selection',digest(s),80,0]]
            with f.open('w',newline='') as stream:csv.writer(stream).writerows(rows)
            a=SimpleNamespace(candidates=c,selection_split=s,scores=f,output=p)
            select(a)
            self.assertEqual(json.loads((p/'selected.json').read_text())['selected']['source'],1)
            rows[1][3]='test'
            with f.open('w',newline='') as stream:csv.writer(stream).writerows(rows)
            with self.assertRaises(ValueError):select(a)
    def test_split_disjoint_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'images.csv';rows=[]
            for i in range(8):
                path=p/f'{i}.bin';path.write_bytes(bytes([i]));rows.append([str(path),i//4])
            def save():
                with f.open('w',newline='') as stream:csv.writer(stream).writerows([['path','label']]+rows)
            save();a=SimpleNamespace(images=f,output=p,seed=42,fraction=.5)
            split(a)
            sets=[{r['sha256'] for r in json.loads((p/(role+'.json')).read_text())['images']}
                  for role in ['selection','test']]
            self.assertFalse(sets[0]&sets[1]);self.assertEqual(len(sets[0]|sets[1]),8)
            rows.append(rows[0]);save()
            with self.assertRaises(ValueError):split(a)

if __name__=='__main__':unittest.main()
