"""Exercise the actual export cells with tiny synthetic fixtures in a temp dir.

This is a software test, never a source of reported patient results.
"""
import contextlib
import io
import json
import platform
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
from PIL import Image

nb = json.loads((Path(__file__).parent / 'kaggle_kernel/brain-tumor-seg-brats-africa.ipynb').read_text(encoding='utf-8'))
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    selected = pd.DataFrame([{'case_id':'fixture', 'cohort':'test-only'}])
    selected.to_csv(root / 'selected_cases.csv', index=False)
    model = {'unet3d':torch.nn.Identity(), 'unetr':torch.nn.Identity()}
    keys = ['dice_ET','dice_TC','dice_WT','dice_mean','hd95_ET','hd95_TC','hd95_WT','hd95_mean']
    record = {'case_id':'fixture', 'cohort':'test-only'}
    for name in model:
        record.update({f'{k}_{name}':0.5 if k.startswith('dice') else 2. for k in keys})
        record[f'time_{name}_s'] = 1.
    image = np.random.default_rng(42).normal(size=(4,24,24,12)).astype(np.float32)
    masks = np.zeros((3,24,24,12), dtype=bool)
    masks[2,5:20,5:20,2:10] = True
    masks[1,8:17,8:17,3:9] = True
    masks[0,10:15,10:15,4:8] = True
    env = dict(np=np, pd=pd, torch=torch, Path=Path, json=json, platform=platform,
               Dict=dict, DEVICE=torch.device('cpu'), selected=selected, SEED=42,
               MODEL=model, META={n:{'sha256':'fixture-only'} for n in model},
               RESULTS=pd.DataFrame([record]),
               pred_cache={'fixture':{'__image__':image,'__gt__':masks,
                                      'unet3d':masks, 'unetr':masks}})
    for index in [16,20,21,22,25]:
        source = ''.join(nb['cells'][index]['source']).replace('/kaggle/working', root.as_posix())
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, f'cell-{index}', 'exec'), env)
    success = json.loads((root / 'SUCCESS.json').read_text())
    assert success['model_case_evaluations'] == 2
    with Image.open(root / 'figures/fixture.gif') as gif:
        assert gif.n_frames > 1
    assert 'fixture' in (root / 'report.html').read_text(encoding='utf-8')
    print('PASS: actual results, figure, animation, chart, HTML export and completion cells')
