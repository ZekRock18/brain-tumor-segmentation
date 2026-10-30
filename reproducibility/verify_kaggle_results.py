"""Check downloaded patient results and report artifacts from a completed run."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=Path)
args = parser.parse_args()
root = args.folder
success = json.loads((root / 'SUCCESS.json').read_text())
metadata = json.loads((root / 'run_metadata.json').read_text())
table = pd.read_csv(root / 'brats_africa_results.csv')
selected = pd.read_csv(root / 'selected_cases.csv')
assert len(table) == success['model_case_evaluations'] == 2 * len(selected)
assert set(table.case_id) == set(selected.case_id) == set(metadata['cases'])
assert not table.duplicated(['case_id', 'model']).any()
assert set(table.model) == {'unet3d', 'unetr'}
for _, rows in table.groupby('case_id'):
    assert set(rows.model) == {'unet3d', 'unetr'}
metrics = [c for c in table if c.startswith(('dice_', 'hd95_'))]
assert np.isfinite(table[metrics].to_numpy()).all()
assert table[[c for c in metrics if c.startswith('dice_')]].ge(0).all().all()
assert table[[c for c in metrics if c.startswith('dice_')]].le(1).all().all()
assert table[[c for c in metrics if c.startswith('hd95_')]].ge(0).all().all()
hashes = {
    'unet3d':'6af7c6c0b81d1219e8729475408bec79d7e44d2018562f40be8f94e4c46ee685',
    'unetr':'0ed2e169b9c2501c6114d87dce5dcd4789eda28b0ace12699435d69aa4f9b32a',
}
for model, digest in hashes.items():
    assert metadata['weights'][model]['sha256'] == digest, f'Unverified weights: {model}'
for name in success['artifacts']:
    path = root / name
    assert path.is_file() and path.stat().st_size > 0, name
    if path.suffix in {'.png', '.gif'}:
        with Image.open(path) as image:
            image.load()
            assert min(image.size) > 20
            if path.suffix == '.gif':
                assert image.n_frames > 1
executed = []
for file in root.rglob('*.ipynb'):
    notebook = json.loads(file.read_text(encoding='utf-8'))
    errors = [o for c in notebook['cells'] for o in c.get('outputs', []) if o.get('output_type') == 'error']
    assert not errors, f'Notebook execution errors in {file}'
    if any(c.get('execution_count') is not None for c in notebook['cells']):
        assert all(c.get('execution_count') is not None
                   for c in notebook['cells'] if c['cell_type'] == 'code'), file
        assert 'END-TO-END COMPLETE' in json.dumps(
            [c.get('outputs', []) for c in notebook['cells']]), file
        executed.append(str(file))
assert executed, 'Download the executed notebook before marking verification complete'
print('VERIFIED:', len(selected), 'real cases;', len(table), 'model-case evaluations')
print('Device:', metadata['device'], 'GPU:', metadata['gpu'])
print('Artifacts:', len(success['artifacts']), '| Executed notebooks:', executed)
print(table.groupby('model')[metrics].mean().round(4).to_string())
