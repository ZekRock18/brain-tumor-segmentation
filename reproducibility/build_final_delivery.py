"""Assemble the verified Kaggle delivery and its three presentation PDFs.

Run from the original workspace. In the packaged reproducibility directory,
run with --documents-only to regenerate the PDFs and index from saved results.
No credentials or patient input scans are copied.
"""
import argparse
import csv
import hashlib
import html
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.platypus import (BaseDocTemplate, Frame, Image, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--documents-only', action='store_true')
args = parser.parse_args()
FINAL = HERE.parent if HERE.name == 'reproducibility' else HERE / 'FINAL_PROJECT'
SOURCE = HERE / 'kaggle_results' / 'v6'
NAVY = colors.HexColor('#16324f')
TEAL = colors.HexColor('#087f8c')
PALE = colors.HexColor('#edf4f8')
GRAY = colors.HexColor('#526474')
KAGGLE = 'https://www.kaggle.com/code/zekrock/brain-tumor-seg-brats-africa'
GITHUB = 'https://github.com/ZekRock18/brain-tumor-segmentation'
LINKS = [
    ('Our project on GitHub — Prakhar Gupta / ZekRock18', GITHUB),
    ('Public Kaggle notebook (packaged results: version 6)', KAGGLE),
    ('Original model repository', 'https://github.com/MohanadMahran/Brain_Tumor_Benchmark'),
    ('UNet3D checkpoint', 'https://huggingface.co/MohanadKombar/brain-tumor-unet3d-brats'),
    ('UNETR checkpoint', 'https://huggingface.co/MohanadKombar/brain-tumor-unetr-brats'),
    ('BraTS-Africa input dataset', 'https://www.kaggle.com/datasets/yakunokamizen/brats-africa'),
    ('Attached checkpoint dataset', 'https://www.kaggle.com/datasets/zekrock/brats-weights'),
]
TEAM = 'Prakhar Gupta | Section F | Roll No. 2430010465<br/>Aditya Dubey | Section F | Roll No. 2430010453'
PROJECT_CREDIT = ('<b>Project owner:</b> Prakhar Gupta (ZekRock18). Project team: Prakhar Gupta and Aditya Dubey. '
    'Our work covers the Kaggle integration, data adaptation, pretrained evaluation, visual reporting and verified delivery. '
    'Original model architectures and pretrained checkpoints are credited separately.')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

if not args.documents_only:
    FINAL.mkdir(exist_ok=True)
    shutil.copytree(SOURCE, FINAL / 'results', dirs_exist_ok=True)
    (FINAL / 'notebooks').mkdir(exist_ok=True)
    for name in ['brain-tumor-seg-brats-africa.ipynb', 'kernel-metadata.json']:
        shutil.copy2(HERE / 'kaggle_kernel' / name, FINAL / 'notebooks' / name)
    repro = FINAL / 'reproducibility'
    repro.mkdir(exist_ok=True)
    for name in ['build_kaggle_notebook.py', 'validate_kaggle_notebook.py',
                 'validate_notebook_exports.py', 'verify_kaggle_results.py',
                 'build_final_delivery.py']:
        shutil.copy2(HERE / name, repro / name)
    (repro / 'kaggle_kernel').mkdir(exist_ok=True)
    for name in ['brain-tumor-seg-brats-africa.ipynb', 'kernel-metadata.json']:
        shutil.copy2(HERE / 'kaggle_kernel' / name, repro / 'kaggle_kernel' / name)
    shutil.copytree(HERE / 'repo', repro / 'repo', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', '.venv',
                        '.pytest_cache', 'gifs', 'outputs', '*.pyc'))
    (FINAL / 'weights').mkdir(exist_ok=True)
    for model in ['unet3d', 'unetr']:
        shutil.copy2(HERE / 'weights' / f'{model}_pytorch_model.bin', FINAL / 'weights')

RESULTS = FINAL / 'results'
df = pd.read_csv(RESULTS / 'brats_africa_results.csv')
meta = json.loads((RESULTS / 'run_metadata.json').read_text())
success = json.loads((RESULTS / 'SUCCESS.json').read_text())
means = df.groupby('model').mean(numeric_only=True)
names = {'unet3d': 'UNet3D', 'unetr': 'UNETR'}
for model in names:
    assert digest(FINAL / 'weights' / f'{model}_pytorch_model.bin') == meta['weights'][model]['sha256']

fontroot = Path('C:/Windows/Fonts')
if (fontroot / 'arial.ttf').exists():
    for name, file in [('Body', 'arial.ttf'), ('BodyBold', 'arialbd.ttf'), ('BodyItalic', 'ariali.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fontroot / file)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='BodyItalic', boldItalic='BodyBold')
else:
    pdfmetrics.registerFontFamily('Helvetica', normal='Helvetica', bold='Helvetica-Bold', italic='Helvetica-Oblique')
FONT = 'Body' if 'Body' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
BOLD = 'BodyBold' if FONT == 'Body' else 'Helvetica-Bold'
styles = {
    'title': ParagraphStyle('Title', fontName=BOLD, fontSize=25, leading=30, textColor=NAVY, spaceAfter=14),
    'subtitle': ParagraphStyle('Subtitle', fontName=FONT, fontSize=13, leading=18, textColor=TEAL, spaceAfter=14),
    'h': ParagraphStyle('Heading', fontName=BOLD, fontSize=14, leading=18, textColor=NAVY, spaceBefore=12, spaceAfter=8, keepWithNext=True),
    'body': ParagraphStyle('BodyText', fontName=FONT, fontSize=10.5, leading=15, textColor=NAVY, spaceAfter=9),
    'small': ParagraphStyle('Small', fontName=FONT, fontSize=8.6, leading=12, textColor=GRAY, spaceAfter=7),
    'cell': ParagraphStyle('Cell', fontName=FONT, fontSize=9, leading=12, textColor=NAVY),
    'headcell': ParagraphStyle('HeadCell', fontName=BOLD, fontSize=9, leading=12, textColor=colors.white),
}

def p(text, kind='body'):
    return Paragraph(text, styles[kind])

def section(story, title, text):
    story.extend([p(title, 'h'), p(text)])

def page(story, title, wide=False):
    if story:
        story.extend([NextPageTemplate('wide' if wide else 'portrait'), PageBreak()])
    story.append(p(title, 'title'))

def table(headers, rows, widths=None):
    cells = [[p(str(x), 'headcell') for x in headers]]
    cells += [[p(html.escape(str(x)), 'cell') for x in row] for row in rows]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#c9d7e2')),
    ]))
    return t

def picture(path, width, maxheight=1000):
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(width / w, maxheight / h)
    return Image(str(path), width=w * scale, height=h * scale, hAlign='CENTER')

def pipeline():
    """Vector explanatory diagram, not a generated medical result."""
    d = Drawing(510, 215)
    labels = [('4 MRI modalities', 'T1 / T1ce / T2 / FLAIR'),
              ('Normalize each scan', 'Clip + brain-only z-score'),
              ('Load pretrained models', 'Strict keys + weight hashes'),
              ('Predict 3D windows', '128 cubed / 25% overlap'),
              ('Blend and threshold', 'Gaussian / sigmoid > 0.5'),
              ('Compare with labels', 'Dice + HD95 + figures')]
    for i, (title, sub) in enumerate(labels):
        col, row = i % 3, i // 3
        x, y = col * 173, 125 - row * 110
        d.add(Rect(x, y, 164, 70, rx=8, ry=8, fillColor=PALE, strokeColor=TEAL))
        d.add(String(x + 82, y + 43, title, fontName=BOLD, fontSize=9.2, textAnchor='middle', fillColor=NAVY))
        d.add(String(x + 82, y + 23, sub, fontName=FONT, fontSize=8, textAnchor='middle', fillColor=GRAY))
        if col < 2:
            d.add(Line(x + 164, y + 35, x + 173, y + 35, strokeColor=TEAL))
    # Connect row one to row two without obscuring labels.
    d.add(Line(428, 125, 428, 105, strokeColor=TEAL))
    d.add(Line(428, 105, 82, 105, strokeColor=TEAL))
    d.add(Line(82, 105, 82, 85, strokeColor=TEAL))
    for x, y in [(171, 160), (344, 160), (171, 50), (344, 50)]:
        d.add(Polygon([x, y, x-4, y+3, x-4, y-3], fillColor=TEAL, strokeColor=TEAL))
    d.add(Polygon([82, 85, 79, 91, 85, 91], fillColor=TEAL, strokeColor=TEAL))
    return d

def regions_diagram():
    d = Drawing(510, 155)
    for x, y, w, h, color in [(10, 5, 250, 145, '#c9dcf7'), (35, 25, 195, 100, '#bfe8cc'), (65, 42, 115, 55, '#f7c7c5')]:
        d.add(Rect(x, y, w, h, rx=10, ry=10, fillColor=colors.HexColor(color), strokeColor=colors.white))
    d.add(String(122, 65, 'ET', fontName=BOLD, fontSize=14, textAnchor='middle', fillColor=NAVY))
    d.add(String(202, 106, 'TC', fontName=BOLD, fontSize=12, textAnchor='middle', fillColor=NAVY))
    d.add(String(232, 132, 'WT', fontName=BOLD, fontSize=12, textAnchor='middle', fillColor=NAVY))
    for y, t in [(120, 'ET is inside TC.'), (96, 'TC is inside WT.'), (62, 'The regions overlap;'), (42, 'they are scored separately.')]:
        d.add(String(285, y, t, fontName=FONT, fontSize=11, fillColor=NAVY))
    return d

def render(filename, title, story):
    def decorate(canvas, doc):
        w, h = canvas._pagesize
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(2)
        canvas.line(42, h - 34, w - 42, h - 34)
        canvas.setFont(FONT, 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(42, 24, 'BraTS-Africa | Pretrained inference | Verified Kaggle v6')
        canvas.drawRightString(w - 42, 24, str(doc.page))
        canvas.restoreState()
    doc = BaseDocTemplate(str(FINAL / filename), pagesize=A4,
        title=title, author='Prakhar Gupta and Aditya Dubey',
        subject='Verified pretrained brain tumor segmentation demonstration; no training',
        leftMargin=42, rightMargin=42, topMargin=50, bottomMargin=43)
    templates = []
    for name, size in [('portrait', A4), ('wide', landscape(A4))]:
        w, h = size
        templates.append(PageTemplate(id=name, pagesize=size, onPage=decorate,
            frames=[Frame(42, 43, w - 84, h - 93, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]))
    doc.addPageTemplates(templates)
    doc.build(story)

def source_block(story):
    for label, url in LINKS:
        story.append(p(f'<b>{label}</b><br/><link href="{url}" color="#087f8c">{url}</link>', 'small'))

metricrows = [[names[m]] + [f'{means.loc[m, k]:.4f}' for k in ['dice_ET', 'dice_TC', 'dice_WT', 'dice_mean', 'hd95_mean']] for m in names]
caserows = []
for case in meta['cases']:
    part = df[df.case_id == case].set_index('model')
    caserows.append([case.replace('BraTS-SSA-', ''), part.iloc[0]['cohort'],
        f"{part.loc['unet3d', 'dice_mean']:.4f}", f"{part.loc['unetr', 'dice_mean']:.4f}",
        f"{part.loc['unet3d', 'hd95_mean']:.2f}", f"{part.loc['unetr', 'hd95_mean']:.2f}"])

# Teacher-facing report: measured results first, then traceable method and figures.
s = []
page(s, '3D Brain Tumor\nSegmentation'.replace('\n', '<br/>'))
s += [p('Pretrained UNet3D and UNETR on real BraTS-Africa MRI', 'subtitle'), p(TEAM),
      p('Final project report | 30 October 2026', 'small'),
      p(f'Public project: <link href="{GITHUB}" color="#087f8c">{GITHUB}</link>', 'small')]
section(s, 'What this project demonstrates',
    'Two released pretrained neural networks segment tumor regions in real three-dimensional brain MRI. '
    'The completed Kaggle run processes four patients with both models, compares their predictions with supplied segmentation labels, '
    'and exports numerical results, comparison figures and animated slice sequences. No training or fine-tuning was performed.')
s.append(table(['Execution', 'Verified outcome'], [
    ['Platform', 'Kaggle version 6; Tesla T4 GPU; CUDA inference'],
    ['Coverage', '4 patients × 2 models = 8 model-case evaluations'],
    ['Evidence', '15 executed code cells; no notebook execution errors; completion marker and 19 expected report artifacts'],
    ['Checkpoint integrity', 'Both models loaded strictly; checkpoint SHA-256 hashes recorded'],
], [125, 386]))
section(s, 'Measured results from this run', 'Dice measures region overlap (higher is better). HD95 measures boundary distance in millimetres (lower is better). Values below are means across the four patients and, for the mean columns, the three tumor regions.')
s.append(table(['Model', 'ET Dice', 'TC Dice', 'WT Dice', 'Mean Dice', 'HD95 mm'], metricrows, [91, 79, 79, 79, 91, 92]))
s.append(p('These are our saved inference results, not the model author\'s validation scores. The difference in mean Dice is small; four cases cannot establish a general model ranking.', 'small'))

page(s, 'Method and scope')
section(s, '1. Real MRI data', 'The attached BraTS-Africa dataset yielded 146 complete cases during discovery. A seeded sample (seed 42) selected four cases, comprising two in the Glioma cohort and two in Other Neoplasms. This composition arose from sampling; it was not a stratified study. Each case contains T1, contrast-enhanced T1, T2, FLAIR and a reference segmentation. The sampled image tensors are 4 × 240 × 240 × 155.')
section(s, '2. Preprocessing and label conversion', 'Each modality is normalized using the repository\'s positive-intensity brain mask, 0.5th/99.5th percentile clipping and brain-only z-score normalization. Background voxels are set to zero. The implementation remaps label 3 to 4, then creates overlapping masks: enhancing tumor (ET) = 4; tumor core (TC) = 1 or 4; whole tumor (WT) = 1, 2 or 4.')
s.append(p('The mapping follows the repository convention. Independent confirmation of annotation semantics against dataset documentation remains a limitation; the scores should be read with that qualification.', 'small'))
section(s, '3. Released pretrained models', 'UNet3D uses a convolutional encoder-decoder with skip connections; UNETR combines a transformer encoder with a convolutional decoder. The architectures and checkpoints come from the credited original project. Strict state-dictionary loading passed for 19,992,603 UNet3D parameters and 19,327,907 UNETR parameters.')
section(s, '4. Full-volume inference', 'Each model predicts on overlapping 128 × 128 × 128 patches, with 25% overlap and Gaussian blending. Sigmoid outputs are thresholded at 0.5. Evaluation mode and disabled gradients prevent learning. CUDA mixed precision is used on the GPU. Metrics are computed on complete three-dimensional volumes, not only the displayed slices.')
section(s, '5. Evidence and deliverables', 'The saved executed notebook, per-region CSV, per-cohort CSV, checkpoint hashes and execution log support the reported run. Four comparison figures, four modality previews and four animated slice sequences accompany an offline HTML report. The animations traverse axial slices through each volume; they are not 3D surface renderings.')

page(s, 'From MRI to segmentation')
s.append(pipeline())
s.append(p('Workflow diagram: the same normalized input is evaluated by both pretrained models. The reference label is used for scoring and visualization, not to update model weights.', 'small'))
s.append(picture(RESULTS / 'figures' / f"{meta['cases'][0]}_modalities.png", 511, 220))
section(s, 'Reading the four input images', 'These are the actual modality previews for BraTS-SSA-00047-000. They show different contrasts of the same patient anatomy. The input channels are combined by the model; the tumor prediction is not obtained by simply coloring bright pixels. The supplied reference segmentation is a separate file used for comparison.')
section(s, 'Why the output is useful to inspect', 'A label map assigns a predicted region membership to each voxel. The overlays make the spatial agreement and missed or excess regions visible. The quantitative evaluation summarizes full-volume agreement, while the images help explain individual cases. Neither replaces a clinical assessment.')

page(s, 'Results and interpretation')
s.append(table(['Case suffix', 'Cohort', 'UNet3D Dice', 'UNETR Dice', 'UNet3D HD95', 'UNETR HD95'], caserows, [90, 113, 77, 77, 77, 77]))
s.append(p('Every entry is a mean over ET, TC and WT for that patient. HD95 is in millimetres. Full case IDs begin with BraTS-SSA-.', 'small'))
s.append(picture(RESULTS / 'dice_comparison.png', 470, 255))
section(s, 'What we can conclude', 'Both pretrained models produced full-volume segmentations on all four real patients. UNETR has slightly higher mean Dice in this sample and lower overall mean HD95; each model has higher mean Dice on two patients. UNet3D has higher average WT Dice. Region-specific and patient-specific differences matter more than declaring an overall winner from this small sample.')
section(s, 'Limits and attribution', 'This is a small illustrative evaluation, not a clinical validation or reproduction of the original training benchmark. Training-set overlap has not been independently excluded, so the patients are not claimed to be guaranteed unseen. No training curves, fine-tuning gains or peak-memory results are claimed. Our contribution is the Kaggle integration, data adaptation, evaluation, visual reporting and execution verification; model design and pretrained training belong to the original authors.')

for i in [0, 2]:
    page(s, f'Patient comparisons {i + 1}–{i + 2}', wide=True)
    s.append(p('Left to right: T1ce MRI, reference segmentation, UNet3D prediction, UNETR prediction. Red: ET; green: core outside ET; blue: whole tumor outside core. Displayed slice has maximal reference WT area. Metrics describe the full volume.', 'small'))
    for case in meta['cases'][i:i+2]:
        s.append(picture(RESULTS / 'figures' / f'{case}.png', 758, 186))
        s.append(p(f'{case} | See results/figures/{case}.gif for the axial slice animation.', 'small'))

page(s, 'Full regional results')
detailrows = [[r.case_id.replace('BraTS-SSA-', ''), names[r.model]] + [f'{getattr(r, k):.4f}' for k in ['dice_ET', 'dice_TC', 'dice_WT']] for r in df.itertuples()]
s.append(table(['Case suffix', 'Model', 'ET Dice', 'TC Dice', 'WT Dice'], detailrows, [120, 100, 97, 97, 97]))
s.append(Spacer(1, 18))
detailrows = [[r.case_id.replace('BraTS-SSA-', ''), names[r.model]] + [f'{getattr(r, k):.2f}' for k in ['hd95_ET', 'hd95_TC', 'hd95_WT']] for r in df.itertuples()]
s.append(table(['Case suffix', 'Model', 'ET HD95', 'TC HD95', 'WT HD95'], detailrows, [120, 100, 97, 97, 97]))
s.append(p('HD95 distances are in millimetres. Values are rounded for presentation; results/brats_africa_results.csv contains full precision. Every case and both models are included, not only the strongest examples.', 'small'))

page(s, 'Sources and evidence')
section(s, 'Where to inspect the completed run', 'Open results/report.html for the offline visual report. Open results/executed.ipynb to inspect the actual outputs of all executed cells. Use brats_africa_results.csv for full-precision case-by-model metrics and run_metadata.json for settings, hardware and checkpoint identities. The clean notebook in notebooks/ is for rerunning, not evidence of a new execution.')
s.append(p(PROJECT_CREDIT))
source_block(s)
section(s, 'Ownership and data handling', 'Original model code and checkpoint authors are credited through the links above. This delivery does not assign a new license to their artifacts. Retain upstream attribution and follow the respective source and dataset terms. Raw MRI inputs are not redistributed in this folder; they remain attached on Kaggle. Figures are generated outputs from the selected cases.')
render('project_report_teacher.pdf', 'Brain Tumor Segmentation — Final Project Report', s)

# Technical reference: enough detail to reproduce and explain the delivered run.
s = []
page(s, 'Project reference')
s += [p('Implementation, reproducibility and evidence guide', 'subtitle'), p(TEAM),
      p(f'Public project: <link href="{GITHUB}" color="#087f8c">{GITHUB}</link>', 'small')]
section(s, 'Read this package in order', 'Start with START_HERE.html, then the teacher report. For a visual demonstration open results/report.html. For execution evidence open results/executed.ipynb. For a new Kaggle run use the clean notebook and metadata under notebooks/.')
s.append(table(['Location', 'Contents and purpose'], [
    ['Three root PDFs', 'Teacher report; this technical reference; presentation script and viva notes'],
    ['results/', 'Executed notebook, real metrics, all figures/GIFs, HTML report, logs and completion metadata'],
    ['notebooks/', 'Clean inference notebook and Kaggle kernel metadata'],
    ['weights/', 'Both original pretrained checkpoint binaries and provenance'],
    ['reproducibility/', 'Notebook builder, validation helpers, PDF/package builder and upstream source snapshot'],
    ['CONTENTS.csv / MANIFEST_SHA256.csv', 'File inventory and integrity hashes'],
], [155, 356]))
section(s, 'Included and deliberately excluded', 'The delivery includes both trained weight artifacts and the source needed for the inference workflow. It excludes credentials, raw MRI scans, abandoned runs, synthetic demo results, Git history, caches and the original author\'s example GIFs. Upstream training code is retained as reference; it was not run for these results. The original source snapshot excludes empty output placeholders.')

page(s, 'Data and preprocessing')
s.append(table(['Input suffix', 'Meaning', 'Model channel'], [
    ['t1n', 'T1-weighted MRI', '0'], ['t1c', 'Contrast-enhanced T1 MRI', '1'],
    ['t2w', 'T2-weighted MRI', '2'], ['t2f', 'FLAIR MRI', '3'], ['seg', 'Reference label volume', 'Evaluation target only'],
], [125, 265, 121]))
section(s, 'Discovery and selection', 'The notebook recursively locates attached data, groups the four modalities and segmentation by patient identity, and retains complete cases. The saved manifest fixes the evaluated patient set: 00047-000, 00068-000, 00111-000 and 00170-000 (all prefixed BraTS-SSA-). Seed 42 supports repeatability, but the manifest is the definitive record if dataset versions change.')
section(s, 'Normalization', 'For each modality, take voxels with intensity greater than zero, clip intensities to the 0.5th and 99.5th percentiles of that mask, subtract the clipped masked mean and divide by the clipped masked standard deviation plus epsilon. Reset background to zero. There is no per-case model training.')
section(s, 'Targets and nested regions', 'The implemented conversion maps 3 to 4 and defines ET = {4}, TC = {1,4}, WT = {1,2,4}. These are overlapping evaluation regions rather than three mutually exclusive classes. The display paints WT first, then TC, then ET so the red enhancing region remains visible. Independently verifying the dataset\'s label semantics remains an explicit limitation of this delivery.')
section(s, 'Geometry and visual selection', 'The sampled volumes use the full 240 × 240 × 155 grid. Header spacing is passed to the HD95 computation. The report uses an axial slice with maximal ground-truth WT area for illustration; this does not change which voxels are scored. Four modality preview images and animations provide additional views.')

page(s, 'Understand the inputs visually')
s.append(picture(RESULTS / 'figures' / f"{meta['cases'][0]}_modalities.png", 511, 220))
section(s, 'A slice, a voxel and a volume', 'A pixel is one location in a 2D image. A voxel is one location in a 3D image grid. An axial slice is one 2D plane through the volume. A 240 × 240 × 155 scan contains 8,928,000 voxels. Four modality volumes are stacked as channels; they describe the same grid with different intensity contrasts.')
section(s, 'What normalization changes', 'MRI brightness scales can differ. Normalization changes the input intensity distribution while preserving its spatial grid. As a simple arithmetic example, a clipped brain intensity of 140 with masked mean 100 and standard deviation 20 becomes (140 - 100) / 20 = 2. This example is explanatory; it is not a measured intensity from the shown patient.')
s.append(regions_diagram())
s.append(p('Schematic of overlapping region definitions; not a medical image or measured tumor shape. In the overlays, the visible colors are drawn in layers, so the blue portion is WT outside TC and the green portion is TC outside ET.', 'small'))
section(s, 'Why three channels instead of one class', 'The models output a probability-like sigmoid value for each region at each voxel. A voxel can belong to ET, TC and WT simultaneously. Each channel is thresholded independently; the shown pipeline does not add an explicit anatomical nesting correction after thresholding.')

page(s, 'Models and inference')
s.append(table(['Feature', 'UNet3D', 'UNETR'], [
    ['Core design', '3D convolutional encoder-decoder; skip connections', 'Transformer encoder; CNN decoder and skip connections'],
    ['Normalization', 'GroupNorm', 'LayerNorm encoder; GroupNorm in decoder'],
    ['Parameter count', '19,992,603', '19,327,907'],
    ['Input / output', '4 MRI channels / 3 region logits', '4 MRI channels / 3 region logits'],
], [115, 198, 198]))
section(s, 'Checkpoint loading', 'Construct each architecture from its checkpoint configuration and load the state dictionary with strict=True. This rejects missing or unexpected parameter keys. SHA-256 checks identify the exact binary files. Parameter counts alone do not prove the weights are pretrained, and model runtime is not evidence of checkpoint authenticity.')
section(s, 'Sliding-window prediction', 'Set model.eval(), disable gradients, use 128-cubed windows with 25% overlap and batch size one, apply sigmoid to each window output, and blend overlapping probability predictions using Gaussian weights. Threshold at 0.5 to form masks. MONAI is used when available; a tested built-in implementation handles its absence, including padding and cropping.')
section(s, 'GPU and reproducibility', 'The recorded run used a Tesla T4, torch 2.11.0+cu128 and Python 3.13.15. The notebook requires CUDA by default to prevent an accidental slow CPU run. Two visible T4 devices do not imply that both were used: this workflow uses the selected CUDA device and does not implement multi-GPU parallelism. Seed 42 is fixed, but hardware and dependency changes can still affect numerical reproducibility.')
section(s, 'Timings are descriptive', 'The first UNet3D case took 52.54 seconds, while subsequent cases took about 9.6–10.0 seconds. UNETR cases took about 2.2 seconds. Startup effects were not isolated in a controlled benchmark, so these measurements should not be promoted as a general speedup claim. Peak VRAM was not measured.')

page(s, 'Understand the model workflow')
s.append(pipeline())
section(s, 'What UNet3D learns to do', 'The encoder applies small 3D convolution filters to learn local patterns and progressively reduce spatial resolution. The decoder upsamples features to recover a mask at the input resolution. Skip connections bring earlier spatial features into the decoder, helping preserve detail. These feature weights were learned by the original author before this demonstration.')
section(s, 'What UNETR changes', 'The transformer encoder represents image patches as tokens and uses attention to combine information across tokens. The convolutional decoder converts those representations back into dense voxel predictions. This is the repository\'s custom UNETR implementation, not a claim that any MONAI UNETR checkpoint is interchangeable with it.')
section(s, 'Why strict loading and eval mode matter', 'A checkpoint stores learned tensors. The architecture defines where those tensors belong. Strict loading checks structural compatibility, while a SHA-256 hash verifies file identity. model.eval() changes training-dependent behavior such as dropout. Disabling gradients avoids building a training graph. Neither step trains the model.')
section(s, 'Why patches overlap', 'A model sees less surrounding context near patch edges. Overlap lets several windows contribute to the same location, and Gaussian weighting gives more influence to the window interior. The combined probability map is cropped back to the original image shape and thresholded before scoring.')

page(s, 'Metrics and interpretation')
section(s, 'Dice coefficient', 'Dice = 2 × |prediction intersection reference| / (|prediction| + |reference|). It quantifies overlap for each of ET, TC and WT. Higher is better, with 1 indicating perfect overlap. The implementation treats two empty masks as Dice 1. A case\'s mean Dice averages its three region scores; the report averages those case values across four patients.')
section(s, '95th-percentile Hausdorff distance', 'HD95 is the 95th percentile of the pooled bidirectional distances between predicted and reference surfaces. Header voxel spacing gives physical distances in millimetres. Lower is better. Two empty masks produce zero. For only one empty mask, the repository uses a fixed nominal-volume diagonal penalty of about 373.13 mm; that special-case penalty is not adapted to each volume\'s spacing.')
rows = [[names[m]] + [f'{means.loc[m, k]:.4f}' for k in ['hd95_ET', 'hd95_TC', 'hd95_WT', 'hd95_mean']] for m in names]
s.append(table(['Model', 'ET HD95', 'TC HD95', 'WT HD95', 'Mean mm'], rows, [115, 99, 99, 99, 99]))
section(s, 'Why both metrics are needed', 'Overlap can be high even when a small misplaced region is far away. For example, case 00170-000 has UNet3D ET Dice 0.9668 but ET HD95 13.63 mm. A good Dice score therefore does not guarantee closely aligned boundaries. Review both masks and metrics.')
section(s, 'Valid claims', 'The run demonstrates successful pretrained inference and evaluation on four labeled MRI volumes. It does not establish clinical usefulness, statistical superiority, robustness across populations, or independently verified out-of-distribution performance. No training/validation split was needed for this inference-only demo; this does not prove the cases were absent from the author\'s training data.')
section(s, 'Original author results are separate', 'Checkpoint validation scores of 0.8694 (UNet3D) and 0.8362 (UNETR) describe the original model artifacts. They are not measurements from this four-case run. Likewise, any training or benchmark tables in the upstream README remain the original author\'s results.')

page(s, 'Worked example: reading a result')
section(s, 'First understand Dice with simple numbers', 'Suppose a prediction contains 100 tumor voxels, the reference contains 120, and 90 are shared. Dice is 2 × 90 / (100 + 120) = 0.8182. This is an illustrative calculation only. Dice is overlap, not classification accuracy and not a probability that the patient has cancer.')
example = df[(df.case_id == meta['cases'][0]) & (df.model == 'unet3d')].iloc[0]
section(s, 'Then read an actual patient', f'For {meta["cases"][0]}, UNet3D has ET Dice {example.dice_ET:.4f}, TC Dice {example.dice_TC:.4f} and WT Dice {example.dice_WT:.4f}. Their arithmetic mean is {example.dice_mean:.4f}. The corresponding mean HD95 is {example.hd95_mean:.2f} mm. Those values come from the full volume, although the figure shows one slice.')
s.append(picture(RESULTS / 'figures' / f"{meta['cases'][0]}.png", 511, 200))
section(s, 'Inspect agreement and disagreement', 'Compare each prediction with the reference on the same slice. Look for missing colored regions, extra colored regions and boundary differences. A single matching slice does not prove good performance across the volume; use the GIF and the full-volume metrics together. Do not infer a clinical diagnosis from the overlay.')
section(s, 'Understand averaging', 'The report treats each of the four patients equally. It averages the three region scores within each patient and then averages across patients. It does not pool all tumor voxels into one global Dice calculation. Large and small tumors can therefore contribute equally to the reported patient-average score.')
section(s, 'Separate completion from scientific validation', 'A success marker and executed cells establish that the pipeline completed. They do not establish correct medical semantics, absence of training-data overlap, or clinical reliability. Those questions require additional dataset documentation and study design.')

page(s, 'How to rerun on Kaggle')
for title, text in [
    ('1. Open or import the notebook', 'Open the public Kaggle notebook using the link listed on the next page, or import notebooks/brain-tumor-seg-brats-africa.ipynb into your own Kaggle account. The saved executed notebook remains an offline backup of version 6.'),
    ('2. Attach the inputs', 'Attach yakunokamizen/brats-africa and zekrock/brats-weights. Access to the latter may depend on the account. If unavailable, create a private checkpoint input from the two binaries in weights/, retaining their names and attribution. Do not upload the entire project folder as data.'),
    ('3. Select a GPU session', 'Complete account phone verification where requested. Select GPU T4 in the notebook accelerator settings and start the session. Confirm the setup cell reports CUDA and a GPU name. An API request for a GPU is not proof of allocation; the executed device output is the evidence.'),
    ('4. Run every cell in order', 'Keep both model and dataset inputs attached. The notebook loads weights, discovers cases, predicts and exports results. The default GPU guard should remain enabled. No training is required. Dependencies and dataset availability can change; inspect any setup failure before continuing.'),
    ('5. Save the completed version', 'Use Save Version / Run All to preserve execution outputs. Check all code cells have run, no errors remain and END-TO-END COMPLETE appears. Confirm SUCCESS.json and all 19 listed artifacts exist. Download outputs and the executed notebook; never interpret a running status as completed results.'),
]:
    section(s, title, text)
section(s, 'Local verification of this delivery', 'From reproducibility/, run: <font name="Courier">python verify_kaggle_results.py ../results</font>. It checks patient/model coverage, finite metrics, expected hashes, readable images, multiframe GIFs, completed notebook cells and the success marker. It requires numpy, pandas and Pillow. No MRI download is needed.')

page(s, 'Provenance and maintenance')
section(s, 'Source snapshot', 'Upstream commit: a76973db5d3c4a437146716bad1236ce400d3c96. The source snapshot under reproducibility/repo retains the original framework for traceability. The notebook builder inlines its architecture, preprocessing and metrics code and adds the Kaggle workflow. The final results are from version 6, not the earlier CPU or synthetic runs.')
section(s, 'Checkpoint identities', 'Full hashes and parameter counts are recorded in weights/provenance.json and results/run_metadata.json. MANIFEST_SHA256.csv covers the packaged files. The manifest excludes itself; CONTENTS.csv supplies a readable inventory. This is an integrity record, not a cryptographic signature from the original author.')
section(s, 'Regenerating documents and code', 'From reproducibility/, run <font name="Courier">python build_final_delivery.py --documents-only</font> to rebuild the PDFs and start page from saved metrics. This needs pandas, Pillow and ReportLab. Run <font name="Courier">python build_kaggle_notebook.py</font> to regenerate the self-contained notebook from the included source. Validation helpers exercise inference/export behavior with synthetic fixtures; those fixtures are not the reported patient results.')
section(s, 'Troubleshooting', 'Missing data: check attached inputs and folder discovery. CUDA guard failure: check account verification, GPU allocation and available quota in the UI. Missing MONAI: the fallback is available, but other imports must still succeed. Strict-load error: check architecture configuration and checkpoint hashes; do not suppress the error. Missing final artifacts: review the execution log and rerun from the failed cell or clean session.')
source_block(s)
s.append(p('No new license is asserted for the upstream code, checkpoints or dataset. Source URLs are attribution and access references; availability and access policies may change.', 'small'))
render('project_reference.pdf', 'Brain Tumor Segmentation — Technical Reference', s)

# Presenter notes: concise, defensible scripts and an offline demonstration path.
s = []
page(s, 'Presentation notes')
s += [p('A practical demonstration and viva guide', 'subtitle'), p(TEAM),
      p(f'Public project: <link href="{GITHUB}" color="#087f8c">{GITHUB}</link>', 'small'),
      p(f'Live notebook: <link href="{KAGGLE}" color="#087f8c">{KAGGLE}</link>', 'small')]
section(s, '30-second introduction', '“Our project evaluates two pretrained 3D brain-tumor segmentation models, UNet3D and UNETR, on real BraTS-Africa MRI scans. We ran both models on four patients using a Kaggle Tesla T4 GPU. The notebook displays MRI scans, reference masks and model predictions, and measures Dice overlap and HD95 boundary distance. We did not train or fine-tune the models.”')
section(s, '90-second explanation', '“Each patient has four MRI modalities that provide complementary information. We normalize them using the original project\'s preprocessing and load its exact pretrained checkpoints. The models process overlapping 3D patches, which are blended into a full-volume prediction. We compare enhancing tumor, tumor core and whole tumor with the supplied labels. Our mean Dice values are 0.8546 for UNet3D and 0.8562 for UNETR. Mean HD95 values are 6.36 and 5.64 millimetres. Both models work on these examples, but four cases are too few to claim one is generally better. Our contribution is making the pretrained workflow run end to end on Kaggle, adapting the data loading and producing verified results and visualizations.”')
s.append(table(['Remember', 'Meaning'], [
    ['4 patients / 8 evaluations', 'Every selected patient was evaluated by both models'],
    ['ET / TC / WT', 'Enhancing tumor / tumor core / whole tumor'],
    ['Dice up; HD95 down', 'More overlap and smaller boundary error are better'],
    ['No training', 'All reported predictions use released pretrained weights'],
], [180, 331]))
section(s, 'Credit the work accurately', 'Say “we integrated and evaluated the pretrained models,” not “we trained these architectures.” Attribute the model code and checkpoints to the original project. Say “four-case demonstration,” not “validated clinical system” or “full benchmark reproduction.”')

page(s, 'Live demonstration sequence')
for title, text in [
    ('1. Show the completed run', 'Open results/executed.ipynb or the saved Kaggle version. Point to the GPU device output, strict weight loading and the final completion marker. Show the saved run rather than starting a fresh GPU execution during the presentation.'),
    ('2. Explain the input', 'Open results/report.html. Show a four-modality preview: T1, contrast-enhanced T1, T2 and FLAIR. Explain that the target is a 3D tumor mask, not a yes/no cancer diagnosis.'),
    ('3. Compare the segmentations', 'Show case 00047-000 first: the same MRI slice is shown with the supplied reference and each model\'s prediction. Explain the red/green/blue overlay legend. Open its GIF to move through the volume. The animation shows changing axial slices, not a rotating 3D mesh.'),
    ('4. Discuss a difficult example', 'Show case 00170-000. UNet3D mean Dice is 0.8611 and UNETR is 0.9381, while their mean HD95 values are 14.18 and 6.75 mm. Use this to explain why visual review and boundary metrics accompany Dice. Other cases favor UNet3D; avoid choosing only flattering examples.'),
    ('5. Close with the actual table', 'Show the four-patient summary and full CSV. State that the very small mean-Dice difference does not establish a general winner. The selected-case manifest, model hashes and executed notebook make the demonstration traceable.'),
]:
    section(s, title, text)
section(s, 'If the internet or GPU is unavailable', 'Open START_HERE.html and results/report.html locally. The PNGs, GIFs, PDFs and CSVs work from the final folder without a new model run. The executed notebook preserves the earlier outputs. Do not claim that an offline report is a new live inference execution.')

page(s, 'Viva questions and honest answers')
qas = [
    ('Why use pretrained models?', 'They already contain learned parameters. The project demonstrates loading, inference and evaluation without the compute and data required to reproduce training.'),
    ('How are the two models different?', 'UNet3D learns volumetric features through convolutions. UNETR uses a transformer encoder with a convolutional decoder. Both consume four MRI channels and output three tumor-region predictions.'),
    ('What proves the checkpoints were used?', 'Strict state-dictionary loading, exact SHA-256 identities and the saved execution outputs. Speed or parameter count alone would not prove this.'),
    ('Why sliding windows?', 'They allow full-size volumes to be processed using manageable 128-cubed patches. Overlap and Gaussian blending combine local predictions across the image.'),
    ('What is ground truth?', 'The dataset\'s supplied reference segmentation. We compare predictions with that annotation; it is not generated by our model.'),
    ('Are these unseen test patients?', 'Training-set overlap has not been independently checked. They are evaluation examples for this demonstration, not a guaranteed unseen test set.'),
    ('Which model wins?', 'UNETR has a slightly higher mean Dice here; each model is better on two cases. Four patients do not support a broad superiority claim.'),
    ('What remains to improve?', 'Confirm label semantics independently, audit overlap with training data, evaluate a larger predefined sample, report uncertainty and benchmark runtime under controlled conditions.'),
    ('Did you implement training?', 'The upstream framework includes training code, retained for reference. Our demonstrated workflow is inference-only and produced no new trained model.'),
]
for q, a in qas:
    s.append(p(f'<b>{q}</b><br/>{a}'))
s.append(p('Before presenting: open the offline report, confirm the four GIFs play, keep the teacher PDF and results table ready, and retain the original final folder structure.', 'small'))
render('presentation_notes.pdf', 'Brain Tumor Segmentation — Presentation Notes', s)

provenance = {'sources': dict(LINKS), 'weights': meta['weights'],
    'notice': 'Original author artifacts; no new license assigned by this delivery.'}
(FINAL / 'weights' / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
buildinfo = {'built_at': datetime.now().isoformat(timespec='seconds'), 'kaggle_version': 6,
    'kaggle_url': KAGGLE, 'notebook_visibility': 'public (confirmed by owner)',
    'github_url': GITHUB, 'project_owner': 'Prakhar Gupta (ZekRock18)',
    'upstream_commit': 'a76973db5d3c4a437146716bad1236ce400d3c96',
    'cases': meta['cases'], 'training': False, 'gpu': meta['gpu'],
    'raw_mri_included': False, 'sources': dict(LINKS),
    'exclusions': ['credentials', 'raw MRI inputs', 'old failed/synthetic runs',
        'Git history and caches', 'upstream example GIFs', 'empty upstream outputs'],
    'regeneration': 'From reproducibility: python build_final_delivery.py --documents-only'}
(FINAL / 'BUILD_INFO.json').write_text(json.dumps(buildinfo, indent=2), encoding='utf-8')
cards = ''.join(f'<a class="card" href="{path}"><strong>{title}</strong><span>{desc}</span></a>' for path, title, desc in [
    ('project_report_teacher.pdf', 'Teacher report', 'Method, measured results, all four patient comparisons and sources'),
    ('project_reference.pdf', 'Technical reference', 'Preprocessing, models, metrics, rerun instructions and provenance'),
    ('presentation_notes.pdf', 'Presentation notes', 'Speaking script, demonstration sequence and viva answers'),
    ('results/report.html', 'Open visual results', 'Offline MRI comparisons, animated slices and result tables'),
    ('results/executed.ipynb', 'Executed notebook', 'Actual saved outputs from completed Kaggle version 6'),
    ('notebooks/brain-tumor-seg-brats-africa.ipynb', 'Rerunnable notebook', 'Clean source for import into Kaggle'),
    ('results/brats_africa_results.csv', 'Full-precision results', 'One row per patient and model; all region scores'),
    ('CONTENTS.csv', 'Complete inventory', 'Every packaged deliverable, its role and size'),
])
rowshtml = ''.join('<tr>' + ''.join(f'<td>{html.escape(v)}</td>' for v in row) + '</tr>' for row in metricrows)
index = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final project | Pretrained brain tumor segmentation</title><style>
body{font:16px/1.65 Arial,sans-serif;background:#f1f5f9;color:#16324f;margin:0}main{max-width:1100px;margin:40px auto;padding:0 24px 50px}
h1{font-size:38px;line-height:1.2}h2{margin-top:35px}.tag{color:#087f8c;font-weight:bold}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
.card{display:block;background:white;padding:22px;border:1px solid #d9e4ee;border-radius:12px;color:#16324f;text-decoration:none}.card:hover{border-color:#087f8c}.card strong{display:block;font-size:19px}.card span{display:block;margin-top:8px;color:#526474}
table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #d9e4ee}th{background:#16324f;color:white}.note{padding:20px;background:#e2f1f0;border-left:4px solid #087f8c}a{color:#087f8c}code{overflow-wrap:anywhere}footer{font-size:13px;margin-top:30px}</style><main>
<div class="tag">FINAL DELIVERY · VERIFIED KAGGLE VERSION 6 · 30 OCTOBER 2026</div>
<h1>Pretrained 3D brain tumor segmentation</h1>
<p>UNet3D and UNETR evaluated on four real BraTS-Africa patients. Eight model-case evaluations completed on a Tesla T4. No training or fine-tuning.</p>
<p>Prakhar Gupta · 2430010465 &nbsp; | &nbsp; Aditya Dubey · 2430010453 &nbsp; | &nbsp; Section F</p>
<p><b>Project by Prakhar Gupta (ZekRock18), with teammate Aditya Dubey.</b> Kaggle integration, data adaptation, pretrained evaluation and visual reporting; original models and checkpoints are credited separately.</p>
<div class="grid">''' + cards + '''</div><h2>Measured results</h2><table><tr><th>Model</th><th>ET Dice</th><th>TC Dice</th><th>WT Dice</th><th>Mean Dice</th><th>Mean HD95 (mm)</th></tr>''' + rowshtml + '''</table>
<p>Dice: higher is better. HD95: lower is better. Means cover the same four patients and three tumor regions.</p>
<div class="note">This is a small illustrative evaluation, not clinical validation. Training-set overlap has not been independently excluded. Label conversion follows the repository convention and requires independent semantic confirmation. Original author validation scores are separate from these measured results.</div>
<h2>Everything in this folder</h2><ul>
<li><b>results/</b>: executed notebook, CSV tables, run metadata, log, completion marker, HTML report, comparison chart, four comparison PNGs, four modality PNGs and four animated slice GIFs.</li>
<li><b>notebooks/</b>: clean inference notebook and Kaggle metadata for rerunning.</li>
<li><b>weights/</b>: both pretrained binaries and provenance, with verified SHA-256 hashes.</li>
<li><b>reproducibility/</b>: notebook/PDF builders, verification helpers and upstream source snapshot. Training code is reference material and was not executed for these results.</li>
<li><b>MANIFEST_SHA256.csv</b>: integrity hashes for package files, excluding the manifest itself. <b>BUILD_INFO.json</b>: version, source and scope record.</li></ul>
<h2>How to use it</h2><ol><li>Read the teacher report and open the visual results.</li><li>Use the presentation notes for the demonstration; the saved results work offline.</li><li>To rerun, import the clean notebook into Kaggle, attach BraTS-Africa and the checkpoint input, select GPU, and follow the technical reference.</li></ol>
<p>Raw MRI scans are not included; they remain on Kaggle. No credentials are included. The notebook is public. Access to the separate weights input may depend on the account; both checkpoint files are also included locally. Do not change relative paths if you want the offline figures to remain linked.</p>
<p><a href="''' + GITHUB + '''">Our public GitHub project</a> · <a href="''' + GITHUB + '''/releases/latest">Download the complete release</a> · <a href="''' + KAGGLE + '''">Open public Kaggle notebook</a> · <a href="https://github.com/MohanadMahran/Brain_Tumor_Benchmark">Original model repository</a></p>
<footer>Original model and checkpoint authors retain attribution and applicable rights. This delivery does not assign a new license to upstream artifacts. Animated GIFs traverse axial slices; they are not surface-rendered 3D reconstructions.</footer></main></html>'''
(FINAL / 'START_HERE.html').write_text(index, encoding='utf-8')

def role(path):
    top = path.parts[0]
    return {'results': 'Verified version 6 execution output', 'notebooks': 'Clean Kaggle rerun source',
        'weights': 'Original pretrained checkpoint / provenance',
        'reproducibility': 'Source and build/verification tooling'}.get(top, 'Final report / navigation / package record')

files = sorted(p for p in FINAL.rglob('*') if p.is_file() and p.name not in {'CONTENTS.csv', 'MANIFEST_SHA256.csv'})
with (FINAL / 'CONTENTS.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f); writer.writerow(['path', 'bytes', 'role'])
    for file in files:
        rel = file.relative_to(FINAL)
        writer.writerow([rel.as_posix(), file.stat().st_size, role(rel)])
    writer.writerow(['CONTENTS.csv', '', 'This inventory; self-size omitted'])
    writer.writerow(['MANIFEST_SHA256.csv', '', 'SHA-256 file integrity manifest'])
with (FINAL / 'MANIFEST_SHA256.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f); writer.writerow(['path', 'bytes', 'sha256'])
    for file in sorted(p for p in FINAL.rglob('*') if p.is_file() and p.name != 'MANIFEST_SHA256.csv'):
        writer.writerow([file.relative_to(FINAL).as_posix(), file.stat().st_size, digest(file)])
print('Built final delivery:', FINAL)
print('Files:', len(list(FINAL.rglob('*'))), '| PDFs: 3 | Real cases:', df.case_id.nunique())
