import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SAMPLER=Path(__file__).resolve().parent/'prepare_route_disjoint_confirmation.py'

class RouteSamplingTests(unittest.TestCase):
    def test_reproducibility_and_all_sibling_instruction_exclusion(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); source=root/'source.json.gz'
            episodes=[{'episode_id':str(2*i+j),'scene_id':'mp3d/a/a.glb','trajectory_id':str(i)} for i in range(5) for j in range(2)]
            with gzip.open(str(source),'wt') as stream: json.dump({'episodes':episodes},stream)
            exclude=root/'old.jsonl'; exclude.write_text(json.dumps({'episode_id':'0','scene_id':'data/scene_datasets/mp3d/a/a.glb'})+'\n')
            cfg={'source':str(source),'scenes':1,'routes_per_scene':3,'selection_seed':9,'experiment_id':'test','exclude_sources':[str(exclude)]}
            outputs=[]
            for name in ['one','two']:
                cfg['output_dir']=str(root/name); path=root/(name+'.json'); path.write_text(json.dumps(cfg))
                subprocess.check_output([sys.executable,str(SAMPLER),'--config',str(path)])
                manifest=json.loads((root/name/'manifest.json').read_text()); outputs.append(manifest)
                self.assertNotIn('0',{e['trajectory_id'] for e in manifest['episodes']})
                self.assertEqual(3,len({e['trajectory_id'] for e in manifest['episodes']}))
            self.assertEqual(outputs[0]['dataset_sha256'],outputs[1]['dataset_sha256'])
            self.assertEqual(outputs[0]['episodes'],outputs[1]['episodes'])
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.check_output([sys.executable,str(SAMPLER),'--config',str(path)],stderr=subprocess.PIPE)
            exclude.write_text('')
            cfg['output_dir']=str(root/'empty'); path.write_text(json.dumps(cfg))
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.check_output([sys.executable,str(SAMPLER),'--config',str(path)],stderr=subprocess.PIPE)
            self.assertFalse((root/'empty').exists())

if __name__=='__main__': unittest.main()
