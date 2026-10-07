import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image
from fia.score_resnet import run
from fia.__main__ import digest


class ResnetScoringTests(unittest.TestCase):
    def test_real_training_zero_bank_and_image_hash(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp); data=p/'shard.pt'; bank=p/'bank.pt'; manifest=p/'selection.json'
                torch.save([torch.rand(4,3,32,32),torch.tensor([0,1,0,1])], data)
                torch.save(torch.zeros(1,3,32,32),bank)
                image=p/'image.png';Image.new('RGB',(32,32),(120,30,90)).save(image)
                manifest.write_text(json.dumps({'role':'selection','images':[{'path':str(image),'label':0,'sha256':digest(image)}]}))
                args=SimpleNamespace(shards=[data],bank=bank,selection_split=manifest,source=0,target=1,classes=2,seeds=[42],epochs=1,lr=.001,batch_size=2,device='cpu',output=p/'out')
                run(args)
                with (args.output/'scores.csv').open() as f: rows=list(csv.DictReader(f))
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['asr'],rows[0]['ftr'])
                self.assertTrue((args.output/'resnet34_seed42.pt').exists())
                Image.new('RGB',(32,32),(121,30,90)).save(image)
                args.output=p/'bad'
                with self.assertRaisesRegex(ValueError,'hash mismatch'):run(args)
                self.assertFalse(args.output.exists())
        finally:
            torch.set_num_threads(previous)
