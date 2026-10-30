"""Build a concise, illustrated beginner-to-project explanation from saved evidence.

Run in the original workspace or from the packaged reproducibility directory.
Uses actual result images; all architecture diagrams are explanatory vectors.
"""
from pathlib import Path
import json
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, PageBreak

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'reproducibility' else HERE / 'FINAL_PROJECT'
RESULTS = ROOT / 'results'
OUT = ROOT / 'understand_the_project.pdf'
NAVY, TEAL, PALE, GRAY = [colors.HexColor(c) for c in ['#16324f', '#087f8c', '#edf4f8', '#526474']]
RED, GREEN, BLUE = [colors.HexColor(c) for c in ['#f7c7c5', '#bfe8cc', '#c9dcf7']]
fontroot = Path('C:/Windows/Fonts')
if (fontroot / 'arial.ttf').exists():
    pdfmetrics.registerFont(TTFont('Guide', str(fontroot / 'arial.ttf')))
    pdfmetrics.registerFont(TTFont('GuideBold', str(fontroot / 'arialbd.ttf')))
    pdfmetrics.registerFontFamily('Guide', normal='Guide', bold='GuideBold', italic='Guide', boldItalic='GuideBold')
    FONT, BOLD = 'Guide', 'GuideBold'
else:
    FONT, BOLD = 'Helvetica', 'Helvetica-Bold'
S = {
    'title': ParagraphStyle('title', fontName=BOLD, fontSize=23, leading=28, textColor=NAVY, spaceAfter=12),
    'sub': ParagraphStyle('sub', fontName=FONT, fontSize=12, leading=17, textColor=TEAL, spaceAfter=12),
    'h': ParagraphStyle('h', fontName=BOLD, fontSize=12.5, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=6, keepWithNext=True),
    'body': ParagraphStyle('body', fontName=FONT, fontSize=10.5, leading=14.5, textColor=NAVY, spaceAfter=8),
    'small': ParagraphStyle('small', fontName=FONT, fontSize=8.5, leading=11.5, textColor=GRAY, spaceAfter=7),
    'cell': ParagraphStyle('cell', fontName=FONT, fontSize=9, leading=12, textColor=NAVY),
    'head': ParagraphStyle('head', fontName=BOLD, fontSize=9, leading=12, textColor=colors.white),
}
story = []
def p(t, kind='body'): return Paragraph(t, S[kind])
def page(n, title, sub):
    if story: story.append(PageBreak())
    story.extend([p(f'{n:02d} / {title}', 'title'), p(sub, 'sub')])
def section(title, text): story.extend([p(title, 'h'), p(text)])
def recall(text):
    t = Table([[p('<b>Say it back:</b> ' + text)]], colWidths=[511])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('BOX',(0,0),(-1,-1),.6,TEAL),
        ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
        ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.extend([Spacer(1,6),t])
def table(headers, rows, widths):
    t = Table([[p(str(x),'head') for x in headers]] + [[p(str(x),'cell') for x in row] for row in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(t)
def photo(name, width=511, height=150):
    path = RESULTS / 'figures' / name
    with PILImage.open(path) as im: w,h=im.size
    scale=min(width/w,height/h)
    story.append(Image(str(path),width=w*scale,height=h*scale,hAlign='CENTER'))
def box(d,x,y,w,h,title,sub='',fill=PALE):
    d.add(Rect(x,y,w,h,rx=7,ry=7,fillColor=fill,strokeColor=TEAL,strokeWidth=.7))
    d.add(String(x+w/2,y+h*.59 if sub else y+h*.43,title,fontName=BOLD,fontSize=9,textAnchor='middle',fillColor=NAVY))
    if sub: d.add(String(x+w/2,y+h*.25,sub,fontName=FONT,fontSize=8,textAnchor='middle',fillColor=GRAY))
def arrow(d,x1,y1,x2,y2,color=TEAL):
    d.add(Line(x1,y1,x2,y2,strokeColor=color,strokeWidth=1))
    if x2>x1: pts=[x2,y2,x2-5,y2+3,x2-5,y2-3]
    elif y2<y1: pts=[x2,y2,x2-3,y2+5,x2+3,y2+5]
    else: pts=[x2,y2,x2-3,y2-5,x2+3,y2-5]
    d.add(Polygon(pts,fillColor=color,strokeColor=color))
def workflow():
    d=Drawing(511,132)
    for i,(a,b) in enumerate([('MRI scans','4 input channels'),('Pretrained models','UNet3D + UNETR'),('Predicted masks','ET / TC / WT')]):
        box(d,i*174,68,163,55,a,b)
        if i<2: arrow(d,i*174+163,95,i*174+174,95)
    box(d,174,0,163,45,'Reference labels','Used only for evaluation')
    box(d,348,0,163,45,'Evaluate outputs','Dice / HD95 / visual report')
    arrow(d,337,22,348,22); arrow(d,429,68,429,45)
    d.add(String(7,18,'No weight updates',fontName=FONT,fontSize=9,fillColor=GRAY))
    return d
def nested():
    d=Drawing(511,115)
    for x,y,w,h,c in [(0,0,245,110,BLUE),(24,16,191,77,GREEN),(50,30,116,43,RED)]:
        d.add(Rect(x,y,w,h,rx=8,ry=8,fillColor=c,strokeColor=colors.white))
    for x,y,t in [(105,47,'ET'),(194,75,'TC'),(224,94,'WT')]:
        d.add(String(x,y,t,fontName=BOLD,fontSize=11,textAnchor='middle',fillColor=NAVY))
    for y,t in [(90,'ET is included in TC.'),(66,'TC is included in WT.'),(32,'These masks overlap;'),(14,'they are not exclusive classes.')]:
        d.add(String(270,y,t,fontName=FONT,fontSize=10,fillColor=NAVY))
    return d
def unet():
    d=Drawing(511,215)
    enc=[(0,156,'128 cubed / 60 ch'),(0,104,'64 cubed / 120 ch'),(0,52,'32 cubed / 240 ch')]
    for x,y,t in enc: box(d,x,y,166,42,'Encoder features',t)
    box(d,174,0,163,42,'Bottleneck','16 cubed / 480 channels')
    for x,y,t in [(345,52,'32 cubed / 240 ch'),(345,104,'64 cubed / 120 ch'),(345,156,'128 cubed / 60 ch')]:
        box(d,x,y,166,42,'Decoder features',t)
    for y in [156,104]: arrow(d,83,y,83,y-10)
    arrow(d,83,52,174,21)
    arrow(d,337,21,427,52)
    for y in [94,146]: arrow(d,428,y,428,y+10)
    for y in [73,125,177]:
        arrow(d,166,y,345,y)
        d.add(String(255,y+5,'copy + concatenate',fontName=FONT,fontSize=8,textAnchor='middle',fillColor=TEAL))
    return d
def transformer():
    d=Drawing(511,190)
    labels=[('128-cubed window','4 MRI channels'),('16-cubed token patches','8 x 8 x 8 = 512 tokens'),('Embedding + position','384 values per token'),
            ('6 transformer blocks','8 attention heads'),('Reshape + CNN decoder','8 -> 16 -> 32 -> 64 -> 128'),('3 output logits','One per tumor region')]
    for i,(a,b) in enumerate(labels):
        row,col=divmod(i,3); x=col*174;y=114-row*100
        box(d,x,y,163,58,a,b)
        if col<2: arrow(d,x+163,y+29,x+174,y+29)
    d.add(Line(429,114,429,94,strokeColor=TEAL));d.add(Line(429,94,81,94,strokeColor=TEAL));arrow(d,81,94,81,72)
    return d
def windows():
    d=Drawing(511,120)
    d.add(String(0,106,'A 240-voxel axis: three 128-voxel windows cover it.',fontName=FONT,fontSize=10,fillColor=NAVY))
    for y,start,label in [(75,0,'0 to 128'),(44,96,'96 to 224'),(13,112,'112 to 240')]:
        d.add(Rect(0,y,480,24,fillColor=PALE,strokeColor=colors.white))
        d.add(Rect(start*2,y,256,24,fillColor=BLUE,strokeColor=TEAL))
        d.add(String(start*2+128,y+8,label,fontName=FONT,fontSize=9,textAnchor='middle',fillColor=NAVY))
    return d

df=pd.read_csv(RESULTS/'brats_africa_results.csv')
means=df.groupby('model').mean(numeric_only=True)
meta=json.loads((RESULTS/'run_metadata.json').read_text())
GITHUB='https://github.com/ZekRock18/brain-tumor-segmentation'
KAGGLE='https://www.kaggle.com/code/zekrock/brain-tumor-seg-brats-africa'

page(1,'The whole idea','Start here: you do not need a machine-learning background.')
story.append(p('<b>Your project:</b> use two already-trained models to mark tumor regions in real brain MRI, compare their masks with reference annotations, and explain the results. This is <b>segmentation</b>: locating regions voxel by voxel, not merely classifying a scan as tumor/no tumor.'))
story.append(workflow())
table(['Term','Plain meaning in this project'],[
    ['AI / machine learning / deep learning','AI is the broad field; machine learning learns patterns from examples; deep learning uses neural networks with many layers.'],
    ['Model / architecture','A model is a numerical mapping from inputs to outputs. Its architecture is the arrangement of operations and layers.'],
    ['Weights / parameters','Learned numbers that control the mapping. Here they come from released checkpoint files.'],
    ['Training / inference','Training changes weights using examples and errors. Inference applies fixed weights to a new input. We ran inference only.'],
    ['Prediction / reference','The model produces a prediction. The dataset supplies a separate annotated reference used to score it.'],
],[123,388])
section('What you actually completed','Four patients × two models = eight evaluations on a Kaggle Tesla T4 GPU. The outputs are full-volume masks, Dice and HD95 scores, comparison images, animations and reports. No training or fine-tuning was performed.')
section('Your contribution and the original authors','Prakhar Gupta (ZekRock18), with teammate Aditya Dubey: Kaggle integration, data adaptation, pretrained evaluation, visualization and verified delivery. Model architectures and pretrained weights are credited to the original Brain_Tumor_Benchmark project and MohanadKombar checkpoints.')
recall('“I built an end-to-end evaluation and visualization workflow around two released pretrained segmentation models.”')

page(2,'Understand the MRI data','One patient is a set of aligned 3D volumes, not one ordinary photograph.')
photo('BraTS-SSA-00047-000_modalities.png',height=138)
story.append(p('Actual input preview from patient 00047. T1, contrast-enhanced T1 (T1ce), T2 and FLAIR are different MRI contrasts of the same anatomy. They provide complementary intensity patterns.', 'small'))
section('Pixels, voxels and channels','A pixel is one location in a 2D picture. A voxel is one location in a 3D grid. An axial slice is one plane through the grid. Each sampled scan is 240 × 240 × 155: 8,928,000 voxels. Stacking four modalities gives an array with shape <b>(4, 240, 240, 155)</b>. The leading 4 means channels, not four patients.')
section('What is a NIfTI file?','A .nii or compressed .nii.gz file stores an image array plus metadata such as voxel spacing and spatial orientation. The image values are intensities; the segmentation file contains integer labels. The suffixes t1n, t1c, t2w and t2f map to T1, T1ce, T2 and FLAIR. seg is the reference mask.')
story.append(nested())
story.append(p('Schematic, not a patient image. ET = enhancing tumor; TC = tumor core; WT = whole tumor. The implementation remaps label 3 to 4, then uses ET={4}, TC={1,4}, WT={1,2,4}. Independent confirmation of dataset annotation semantics remains a limitation.', 'small'))
section('How to read the colors','The overlay paints WT first, TC next and ET last. Red shows ET; the visible green is TC outside ET; visible blue is WT outside TC. However, each complete overlapping mask is evaluated separately. Colors are display choices, not original MRI colors.')
recall('“Each patient has four MRI input channels plus a separate reference label volume. ET, TC and WT are nested tumor regions.”')

page(3,'Prepare the input and weights','The same architecture and preprocessing must match the checkpoint.')
section('1. Find complete patients','The notebook discovers matching files by patient ID and keeps cases with all four modalities and a reference segmentation. Discovery found 146 complete cases. Seed 42 selected four; selected_cases.csv records exactly which ones. They happened to include two Glioma and two Other Neoplasms cases. This was not a stratified clinical study.')
section('2. Normalize intensities','For each modality, define the brain mask as intensity &gt; 0. Clip intensities to the 0.5th and 99.5th percentiles within that mask. Compute the clipped brain mean and standard deviation, then use <b>z = (intensity - mean) / (standard deviation + epsilon)</b>. Set background to zero. This changes intensity scale, not the spatial grid.')
story.append(p('<b>Small example:</b> with intensity 140, mean 100 and standard deviation 20, z = 2. This is an arithmetic illustration, not a measured patient value. Epsilon is a tiny number that avoids division by zero.'))
section('3. Load the learned model','Construct each architecture using its saved configuration, then load its state dictionary with <b>strict=True</b>. The state dictionary maps parameter names to tensors (multidimensional arrays of numbers). Strict loading checks that names and shapes match. A SHA-256 hash identifies the exact checkpoint file. Parameter count or fast execution alone cannot establish checkpoint authenticity.')
table(['Operation','Meaning'],[
    ['model.eval()','Switch training-dependent layers, such as dropout, to evaluation behavior. It does not train or load weights.'],
    ['Disable gradients','Avoid storing the computation graph used for weight updates; reduces inference overhead.'],
    ['GPU + mixed precision','CUDA runs numerical operations on the GPU; mixed precision uses lower precision for suitable operations. It does not mean multiple GPUs are used.'],
],[145,366])
section('What training would have done','Training repeats: predict, calculate a loss (numerical error against labels), backpropagate gradients (how that error changes with each weight), then use an optimizer to update the weights. A batch is the examples processed together; an epoch is one pass through a training set. Fine-tuning continues learning from existing weights. None of these weight-update steps ran here.')
recall('“Our checkpoints already contain learned weights. We normalize scans, load compatible weights strictly and run evaluation mode without updates.”')

page(4,'Model one: UNet3D','Convolutions learn local 3D patterns; the decoder restores spatial detail.')
story.append(unet())
story.append(p('Actual checkpoint widths: 60 → 120 → 240 → 480 channels, not the class default widths. Spatial sizes above apply to one 128-cubed inference window. Each arrow across the diagram is a skip connection.', 'small'))
section('A. Convolution: learn a small pattern detector','A 3 × 3 × 3 filter moves over the volume. At each location it multiplies nearby values by learned weights and sums them, across input channels. Multiple filters produce multiple feature maps, which can represent useful patterns. “Channels” inside the network are learned features, not extra MRI scans.')
section('B. Encoder: smaller grids, richer features','The encoder uses four levels of convolution blocks. Max pooling keeps the largest value in each 2 × 2 × 2 feature neighborhood. Three pooling steps halve the grid: 128 → 64 → 32 → 16. More channels represent richer features at coarser resolutions. Each block uses two convolutions, GroupNorm (4 groups here), LeakyReLU and a residual addition.')
section('C. Decoder: recover where each region belongs','Transposed convolutions upsample 16 → 32 → 64 → 128. The decoder concatenates features from the matching encoder level: these long skip connections restore fine spatial information. More convolution blocks combine the features. A final 1 × 1 × 1 convolution produces three logits per voxel.')
section('The small terms, decoded','<b>Activation:</b> a nonlinear function that enables complex mappings; LeakyReLU keeps a small slope for negative values. <b>GroupNorm:</b> normalizes feature groups within a sample. <b>Residual addition:</b> adds a block input to its transformed output. This differs from the U-Net skip, which concatenates encoder and decoder channels.')
recall('“UNet3D compresses a volume into learned features, then upsamples them. Skip connections preserve location details. Our model has 19,992,603 parameters.”')

page(5,'Model two: UNETR','A transformer shares information among patches; a CNN decoder produces voxel masks.')
story.append(transformer())
story.append(p('Verified checkpoint: token patch size 16, embedding dimension 384, six blocks, eight heads. A 128-cubed window contains (128/16)³ = 512 tokens. This is the repository’s custom UNETR, not a generic interchangeable checkpoint.', 'small'))
section('A. Turn image patches into tokens','A strided 3D convolution projects each 16 × 16 × 16 patch into a vector of 384 numbers: its embedding. Learned position embeddings tell the network where each token belongs. Without position information, the token features alone would not explicitly identify their locations.')
section('B. Self-attention: share relevant information','For each token, learned projections create a <b>query</b> (what information it seeks), a <b>key</b> (what it matches) and a <b>value</b> (information it contributes). Query-key dot products produce scores; softmax converts them into weights that sum to one. A weighted sum of values updates the token. This lets tokens interact across the current window, not automatically across the entire patient volume.')
story.append(p('<b>Formula:</b> Attention(Q,K,V) = softmax(QK<super>T</super> / √d) V.<br/>Q, K and V are learned projections of the same token sequence; d is the number of features per head. Here d = 384/8 = 48. Eight heads can learn different relationships; they are not eight separate models.','small'))
section('C. Repeat and decode','Each of six blocks uses LayerNorm, attention and residual addition, then another LayerNorm and a feed-forward network (384 → 1536 → 384, with GELU activation), plus residual addition. LayerNorm normalizes features within each token. Layer-3 and final layer-6 features feed the decoder, which reshapes tokens to an 8³ grid and upsamples to 128³. Original-input features provide another skip. The CNN decoder uses GroupNorm; the transformer encoder uses LayerNorm.')
recall('“UNETR represents patches as tokens, relates them using attention and decodes the features back to masks. Our model has 19,327,907 parameters.”')

page(6,'From windows to a full mask','There are two different patch sizes: inference windows and UNETR token patches.')
table(['Size','Role'],[
    ['240 × 240 × 155','The full patient volume in this demonstration.'],
    ['128 × 128 × 128','The inference window fed to either model.'],
    ['16 × 16 × 16','The smaller token patch used inside UNETR only.'],
],[155,356])
section('Why not send the whole volume at once?','Large 3D feature maps consume memory. Sliding windows let the model handle manageable pieces while retaining the full patient grid. Neighboring windows overlap by a nominal 25%. The final window is shifted to cover the edge, so its overlap can be larger.')
story.append(windows())
section('Logit → sigmoid → blended probability → mask','Each model returns three raw scores, called logits, at each voxel. Sigmoid maps a logit z to <b>1 / (1 + exp(-z))</b>, between 0 and 1. Each window’s sigmoid output is blended into the full grid. A value greater than 0.5 becomes foreground. These outputs are model scores, not guaranteed calibrated medical probabilities.')
section('Gaussian blending prevents hard joins','For a voxel covered by several windows, compute <b>sum(weight × probability) / sum(weight)</b>. Gaussian weights favor window interiors. For example, predictions 0.8 and 0.4 with weights 0.75 and 0.25 blend to 0.7, then become foreground. This is an illustration, not an actual stored voxel.')
section('Fair comparison and practical details','Both models use the same selected patients, preprocessing, threshold and evaluation procedure. The notebook uses MONAI when available and a tested built-in fallback otherwise. Masks are thresholded independently; no explicit post-processing enforces ET inside TC inside WT. Inference uses the selected T4 device, not a two-GPU distributed run.')
recall('“We predict overlapping 128-cubed windows, Gaussian-blend their sigmoid scores, then threshold the full-volume maps.”')

page(7,'Know what the metrics mean','Dice asks “how much overlap?”; HD95 asks “how far apart are the boundaries?”')
section('Dice: region overlap, from zero to one','<b>Dice = 2 × shared foreground voxels / (predicted foreground + reference foreground).</b> A score of 1 means exact overlap. If a prediction has 100 voxels, the reference 120, and 90 are shared, Dice = 180/220 = 0.8182. This is not 81.82% classification accuracy and not the chance that the patient has a tumor.')
section('Why not ordinary voxel accuracy?','Most voxels may be background. A model that predicts background everywhere could score highly on accuracy while missing the tumor. Dice focuses on foreground overlap. Still, a good Dice can hide important local errors, so we also inspect images and boundary distance.')
section('HD95: boundary distance in millimetres','Extract both mask surfaces. For every predicted surface point, find its nearest reference-surface distance; also calculate the reverse direction. Pool these distances and take the 95th percentile. This is HD95 in this implementation. Lower is better. Header voxel spacing converts grid distances into physical units. It is not the mean of the worst 5% of errors.')
section('Why we report both','Two large masks may overlap strongly yet have a small, distant mismatch. In actual case 00170, UNet3D ET Dice is 0.9668 but ET HD95 is 13.63 mm. High overlap does not guarantee that all boundaries are close. Conversely, percentile-based HD95 can ignore a small fraction of extreme errors.')
table(['Edge case / averaging','Implemented behavior'],[
    ['Both masks empty','Dice = 1; HD95 = 0.'],
    ['Only one mask empty','Dice is zero or near zero with numerical smoothing; HD95 uses the repository’s fixed nominal-volume diagonal penalty, about 373.13 mm.'],
    ['Mean Dice / HD95','Average the three region scores per case, then average across four cases. Every case has equal weight; voxels are not pooled into one global Dice.'],
],[150,361])
recall('“Dice rewards matching regions; HD95 penalizes separated boundaries. I need both metrics and the visual overlays to assess a case.”')

page(8,'Read your actual results','These measurements come from the completed Kaggle run, not a synthetic test.')
rows=[]
for m,label in [('unet3d','UNet3D'),('unetr','UNETR')]:
    rows.append([label]+[f'{means.loc[m,k]:.4f}' for k in ['dice_ET','dice_TC','dice_WT','dice_mean','hd95_mean']])
table(['Model','ET Dice','TC Dice','WT Dice','Mean Dice','HD95 mm'],rows,[86,77,77,77,97,97])
story.append(Spacer(1,8))
photo('BraTS-SSA-00047-000.png',height=133)
story.append(p('Case 00047: MRI → reference → UNet3D → UNETR. Mean Dice: 0.9105 vs 0.8606. The displayed axial slice has the largest reference WT area; scores cover the entire volume.','small'))
photo('BraTS-SSA-00170-000.png',height=133)
story.append(p('Case 00170: mean Dice 0.8611 vs 0.9381; mean HD95 14.18 vs 6.75 mm. Model performance changes with the patient. Open the saved GIFs to inspect more than one slice.','small'))
section('What you can say','Both models produced segmentations for all four cases. UNETR has slightly higher mean Dice and lower mean HD95 in this sample; UNet3D has higher mean WT Dice. Each model has the higher patient-mean Dice on two cases. The overall Dice gap is only about 0.0017, not evidence of a generally superior model.')
section('What the sample cannot prove','Four cases are not clinical validation. Training-set overlap was not independently excluded, so do not promise unseen patients or a proven out-of-distribution benchmark. Label semantics need independent confirmation. The original author’s validation scores, 0.8694 and 0.8362, are different results. No training gains or controlled speed/peak-memory claims are made here.')
recall('“Our measured means are 0.8546 and 0.8562 Dice. The result demonstrates a working pretrained evaluation, not a universal winner.”')

page(9,'Connect the explanation to code','Use this page to narrate the notebook from start to finish.')
table(['Stage','What happens and where to look'],[
    ['1. Setup','Imports libraries, fixes seed 42, checks CUDA. GPU guard stops an unintended CPU run. PyTorch runs models; NumPy handles arrays; nibabel reads NIfTI.'],
    ['2. Model definitions','The notebook contains UNet3D and UNETR classes from the original project. Source: reproducibility/repo/models/.'],
    ['3. Preprocess + metrics','Normalization, region conversion, Dice and HD95. Original source: data/preprocessing.py and training/metrics.py in the source snapshot.'],
    ['4. Load weights','Read checkpoint configuration, construct matching architectures, load strictly and record hashes. A .bin suffix is a saved file, not a model architecture.'],
    ['5. Discover / select','Find all modalities and labels, choose cases and record their identities. Inputs are attached on Kaggle, not downloaded to your laptop.'],
    ['6. Predict / score','Run both models over the full volume through sliding windows; compute ET/TC/WT metrics against the labels.'],
    ['7. Export / verify','Save CSVs, images, animated slices and report.html. SUCCESS.json lists expected artifacts; END-TO-END COMPLETE records successful completion.'],
],[125,386])
section('The four files to understand first','<b>results/executed.ipynb:</b> historical execution and outputs, all 15 code cells completed.<br/><b>results/brats_africa_results.csv:</b> one row per patient and model.<br/><b>results/run_metadata.json:</b> settings, GPU and checkpoint identities.<br/><b>results/report.html:</b> the offline visual demonstration.')
section('How to rerun or present','Use the public Kaggle notebook, attach BraTS-Africa and the checkpoint input, select a GPU and run all cells. Confirm actual CUDA output and final artifacts rather than assuming a GPU request succeeded. For a presentation, open the saved HTML report and PDFs; no live GPU is needed to view existing results. A new run is not implied by viewing the report.')
recall('“The executed notebook proves what ran; the CSV gives exact numbers; metadata explains the settings; the visual report makes the findings understandable.”')

page(10,'Explain it without memorizing','Practice these answers, then close the PDF and explain them in your own words.')
qas=[
    ('What is the project?', 'A reproducible Kaggle demonstration comparing two released pretrained 3D tumor-segmentation models on four real labeled MRI cases.'),
    ('Why four inputs and three outputs?', 'The four inputs are MRI modalities. The three outputs are ET, TC and WT region scores at every voxel; they are not three patient diagnoses.'),
    ('What is the main model difference?', 'UNet3D extracts local volumetric features with convolutions. UNETR relates patch tokens with attention, then uses a CNN decoder to produce dense masks.'),
    ('Did you train either model?', 'No. The original author supplied trained checkpoints. We integrated loading, data handling, inference, evaluation, visualization and verification.'),
    ('Why strict loading and hashes?', 'Strict loading checks structural compatibility. SHA-256 identifies the exact file. Neither runtime nor parameter count alone proves checkpoint identity.'),
    ('Why 128 and 16?', '128 cubed is the window fed to both models. 16 cubed is UNETR’s internal token patch; each window therefore contains 512 tokens.'),
    ('Does a higher Dice mean clinically better?', 'Not by itself. Boundary errors, sample size, label validity, population coverage and clinical validation all matter. This is an educational demonstration.'),
    ('What would you improve next?', 'Confirm label semantics, audit training overlap, evaluate a larger predefined sample, quantify uncertainty and measure runtime under controlled conditions.'),
]
for q,a in qas: story.append(p(f'<b>{q}</b><br/>{a}'))
section('Your 30-second closing explanation','“I use four MRI modalities as input, normalize them, load the exact pretrained UNet3D and UNETR weights, and predict overlapping 3D windows. After blending and thresholding, I compare ET, TC and WT against the supplied labels using Dice and HD95. My four-case run completed on a T4 GPU, with mean Dice 0.8546 and 0.8562. The project includes traceable results and visual reports, while acknowledging the small sample and remaining validation limits.”')
story.append(p(f'<b>Project:</b> <link href="{GITHUB}" color="#087f8c">{GITHUB}</link><br/><b>Kaggle:</b> <link href="{KAGGLE}" color="#087f8c">{KAGGLE}</link>', 'small'))
story.append(p('Evidence: saved run metadata, per-case CSV and exact checkpoint configurations; architecture details checked against reproducibility/repo/models/unet3d.py and unetr.py. Original models: MohanadMahran/Brain_Tumor_Benchmark; pretrained weights: MohanadKombar on Hugging Face. The other project PDFs provide full source links and reproducibility details.','small'))

def footer(canvas, doc):
    canvas.saveState();w,h=A4
    canvas.setStrokeColor(TEAL);canvas.setLineWidth(2);canvas.line(42,h-34,w-42,h-34)
    canvas.setFont(FONT,8);canvas.setFillColor(GRAY)
    canvas.drawString(42,23,'Understand the project | Prakhar Gupta / ZekRock18 | Pretrained inference')
    canvas.drawRightString(w-42,23,f'{doc.page} / 10')
    canvas.restoreState()
doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=50,bottomMargin=43,
    title='Understand the Project: MRI to UNet3D and UNETR Results',author='Prakhar Gupta',
    subject='A concise illustrated beginner guide to the verified BraTS-Africa inference project')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Built learning guide:',OUT)
