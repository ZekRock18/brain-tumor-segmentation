"""Local regression checks; does not download or run real patient scans."""
import ast
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).parent
nb = json.loads((ROOT / 'kaggle_kernel/brain-tumor-seg-brats-africa.ipynb').read_text(encoding='utf-8'))
env = dict(np=np, pd=pd, torch=torch, F=F, Path=Path, Dict=Dict, List=List,
           Optional=Optional, Tuple=Tuple, DEVICE=torch.device('cpu'))
definitions = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    transformed = TransformerManager().transform_cell(source)
    compile(transformed, f'cell-{i}', 'exec')
    for node in ast.parse(transformed).body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            'build_manifest', '_gaussian_kernel', '_starts', 'predict_full_volume',
            'to_gray', 'overlay', 'best_slice_index', 'compute_dice',
            'compute_surface_distances', 'compute_hd95', 'compute_all_metrics',
            'convert_labels_to_regions'}:
            definitions.append(node)
exec(compile(ast.Module(body=definitions, type_ignores=[]), '<functions>', 'exec'), env)
env['MODALITY_SUFFIX'] = {'t1':'-t1n','t1ce':'-t1c','t2':'-t2w','flair':'-t2f','label':'-seg'}
env['REGION_COLORS'] = [(0.20,0.45,1.0),(0.15,0.90,0.25),(1.0,0.15,0.15)]
import warnings
from scipy import ndimage
env.update(warnings=warnings, ndimage=ndimage, REGION_NAMES=['ET','TC','WT'],
           HD95_PENALTY=float(np.sqrt(240**2+240**2+155**2)))

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    for cohort in ['95_Glioma', '51_OtherNeoplasms']:
        case = root / cohort / ('case-' + cohort)
        case.mkdir(parents=True)
        for suffix in env['MODALITY_SUFFIX'].values():
            (case / (case.name + suffix + '.nii')).touch()
    manifest = env['build_manifest'](list(root.rglob('*-seg.nii')))
    assert set(manifest.cohort) == {'95_Glioma','51_OtherNeoplasms'}
    assert len(manifest.dropna()) == 2

class PointwiseModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :3] * 0.7

# Exercise overlapping windows, padded dimensions, and edge voxels against
# an exact pointwise oracle, then compare the MONAI and fallback paths.
model = PointwiseModel()
rng = np.random.default_rng(42)
for shape in [(4, 13, 11, 9), (4, 5, 9, 6)]:
    image = rng.normal(size=shape).astype(np.float32)
    expected = torch.sigmoid(torch.from_numpy(image[:3]) * .7).numpy()
    monai_result = env['predict_full_volume'](model, image, roi=(8,8,8))
    with patch.dict('sys.modules', {'monai.inferers': None}):
        fallback = env['predict_full_volume'](model, image, roi=(8,8,8))
    np.testing.assert_allclose(fallback, expected, atol=2e-6)
    np.testing.assert_allclose(monai_result, expected, atol=2e-6)

# A spatial convolution produces different edge predictions per window,
# so agreement here also checks Gaussian blending and window placement.
torch.manual_seed(7)
spatial_model = torch.nn.Conv3d(4, 3, kernel_size=3, padding=1)
for shape in [(4,13,11,9), (4,5,9,6)]:
    image = rng.normal(size=shape).astype(np.float32)
    reference = env['predict_full_volume'](spatial_model, image, roi=(8,8,8))
    with patch.dict('sys.modules', {'monai.inferers': None}):
        fallback = env['predict_full_volume'](spatial_model, image, roi=(8,8,8))
    np.testing.assert_allclose(fallback, reference, atol=3e-6)

base = np.zeros((3,3), dtype=np.float32)
regions = np.zeros((3,3,3), dtype=bool)
regions[2,0,:] = True
regions[1,0,:2] = True
regions[0,0,0] = True
rgb = env['overlay'](base, regions)
assert rgb[0,0].argmax() == 0, 'ET must be red'
assert rgb[0,1].argmax() == 1, 'TC outside ET must be green'
assert rgb[0,2].argmax() == 2, 'WT outside TC must be blue'
labels = env['convert_labels_to_regions'](np.array([[[0,1,2,3,4]]]))
np.testing.assert_array_equal(labels[:,0,0], [[0,0,0,1,1],[0,1,0,1,1],[0,1,1,1,1]])
a = np.zeros((5,5,5), bool)
b = a.copy()
a[1,2,2] = True
b[2,2,2] = True
assert env['compute_hd95'](a, b, (2.,1.,1.)) == 2.
assert env['compute_dice'](a, a) > .999999
print('PASS: notebook cell compilation, nested manifest, padded/overlapping inference, overlay colors, label mapping, physical HD95')
