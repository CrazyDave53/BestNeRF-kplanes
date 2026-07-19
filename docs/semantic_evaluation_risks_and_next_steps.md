# Semantic Evaluation Risks And Next Steps

This note summarizes the critique in the attached thesis-review text and turns it
into a practical experiment backlog for the BestNeRF/K-Planes semantic branch.

Source note:
`C:\Users\minhc\.codex\attachments\28acd3a7-239a-4ff8-9e0d-91fda4d4df49\pasted-text.txt`

## Current Status

We have a working semantic branch:

- RGB baseline smoke passed on Kaggle.
- Semantic smoke passed on Kaggle.
- Full semantic training passed for Coffee Martini:
  - 10,000 steps.
  - RGB PSNR around 30.
  - semantic loss around 0.03.
- Semantic checkpoint is persisted as Kaggle dataset:
  - `lenguyenminhchau/coffee-martini-kplanes-semantic-ds16-64f`
- OpenSeg cache is persisted as Kaggle dataset:
  - `lenguyenminhchau/coffee-martini-openseg-ds16-64f`
- A script exists to render a fixed `human` query heatmap:
  - `scripts/render_semantic_query_neu3d.py`
- A first reviewed model-assisted human annotation set exists:
  - `lenguyenminhchau/coffee-martini-human-annotations-cam01-64f`
  - scope: `cam01`, frames `0,8,16,24,32,40,48,56`, queries `human` and
    `hand`
- Human-mask evaluation exists for both `student_kplanes` and
  `teacher_openseg` against those reviewed masks:
  - `human`: student AP `0.9657`, best IoU `0.9183`
  - `human`: teacher AP `0.9518`, best IoU `0.8917`
  - `hand`: both teacher and student are weak, so this query needs prompt or
    target reconsideration.

The code working is a strong engineering milestone. It is not yet enough for a
strong scientific claim.

## Main Evaluation Risk

The strongest critique is that the current setup can only prove this narrow
claim:

> Dynamic K-Planes can learn to preserve or imitate OpenSeg heatmaps on the
> Coffee Martini scene.

It does not yet prove these stronger claims:

- The model performs correct semantic segmentation.
- The model understands open-vocabulary 4D semantics.
- The model tracks objects consistently through time.
- The method generalizes beyond one dynamic scene.
- The method beats strong open-vocabulary 3D/4D baselines.

The reason is simple: OpenSeg is both teacher and evaluation target. If OpenSeg
is wrong, the student can imitate the wrong teacher and still score well.

## Critical Issues To Avoid In The Thesis

### 1. Claims Stronger Than Evidence

Avoid saying the model "segments correctly" or "achieves high semantic
accuracy" unless we have independent human ground truth masks.

Safer claim:

> The model distills OpenSeg features into a dynamic K-Planes representation and
> preserves OpenSeg-like query heatmaps on Coffee Martini.

### 2. Teacher-Student Evaluation Loop

Current semantic metrics compare the student against OpenSeg. This measures
teacher imitation, not human-label correctness.

Needed:

- Add a small human-labeled mask set.
- Report metrics against those masks separately from teacher-imitation metrics.

### 3. No Ground Truth Semantic Masks

Without manual masks, query heatmaps for `human`, `cup`, `glass`, `hand`,
`coffee`, etc. are only indirectly validated.

Needed:

- Manually annotate a small set of frames and queries.
- Compute binary mask metrics such as IoU, mIoU, AP, precision, recall, and F1.

### 4. Only One Scene

Coffee Martini alone is too narrow for a general 4D semantics claim.

Needed:

- Add more Neu3D scenes if time allows.
- At minimum, state clearly that the result is demonstrated on one scene.

### 5. Missing RGB Reconstruction Evidence

If the thesis claims high rendering quality, report RGB metrics:

- PSNR.
- SSIM.
- LPIPS.
- Training time.
- GPU memory.
- Render time or FPS.

### 6. Missing Ablations

The critique calls out missing evidence for:

- targeted sampling;
- semantic rendering strategy;
- top-k choice;
- temporal consistency;
- static/dynamic split;
- Temporal-TV;
- loss choice.

## Best Experiments To Try Next

### Experiment 1: Render Human Query Heatmaps

Goal:
Verify that the trained semantic branch produces interpretable output.

Status:
Done for the current checkpoint. The `human` visualization looked good enough
to move on to annotation and metrics.

Run the existing script on several frames:

```bash
cd /kaggle/working/BestNeRF/k-planes

CKPT=/kaggle/working/BestNeRF/k-planes/logs/baseline/cm_semantic_64f_ds16/model.pth

for IDX in 0 1 2 3 4 5 6 7; do
  PYTHONPATH=. python scripts/render_semantic_query_neu3d.py \
    --config-path plenoxels/configs/local/dynerf_cm_semantic_64f_ds16.py \
    --checkpoint "$CKPT" \
    --output-dir "/kaggle/working/human_query/frame_${IDX}" \
    --frame-index "$IDX" \
    --batch-size 2048 \
    --amp
done
```

Inspect:

```bash
find /kaggle/working/human_query -type f -name 'human_overlay.png' -print
```

Outputs per frame:

- `rgb.png`
- `human_heatmap_gray.png`
- `human_heatmap_color.png`
- `human_overlay.png`
- `human_scores.npy`

What to look for:

- Does the heatmap activate on the person/hands/body?
- Does it activate on background objects incorrectly?
- Is it temporally stable across adjacent frames?

### Experiment 2: Compare Student Heatmap To OpenSeg Teacher

Goal:
Quantify whether the model preserves the teacher heatmap.

This is not ground truth semantics, but it is useful as a teacher-imitation
metric.

Metrics:

- Pearson correlation.
- Spearman correlation.
- NDCG.
- AP after thresholding teacher heatmap.
- AUIoU over multiple thresholds.

Try this for:

- `human`;
- `cup`;
- `glass`;
- `hand`;
- `coffee`;
- `martini`;
- `table`.

Expected scientific claim:

> The model preserves teacher query heatmaps.

Not:

> The model segments objects correctly.

### Experiment 3: Model-Assisted Human Annotation Set

Goal:
Break the teacher-student evaluation loop.

We will use a strong segmentation model, such as Grounded-SAM2, to propose
initial masks, but the benchmark should be described as human annotated because
every mask is reviewed and corrected or rejected by a human before inclusion.

Thesis wording:

> We construct a human-annotated evaluation set using a model-assisted
> annotation workflow. Grounded-SAM2 proposes initial masks, and every mask is
> manually reviewed and corrected or rejected before inclusion.

Small but valuable first dataset:

- 5 to 10 frames.
- 1 to 3 camera views.
- 2 to 4 queries.

Start with:

- `human`
- `hand`
- `glass`
- `table`

Suggested annotation layout:

```text
coffee_martini_human_annotations/
  manifest.csv
  images/
    cam00_frame000.png
  masks/
    human/
      cam00_frame000.png
  overlays/
    human/
      cam00_frame000_overlay.png
```

Suggested manifest columns:

```text
image_path,proposal_path,mask_path,overlay_path,query,camera,frame,status,source,reviewer,notes
```

Accepted statuses:

- `accepted`
- `corrected`
- `rejected`
- `ambiguous`

Only `accepted` and `corrected` masks should be used for metrics.

Preparation command:

```bash
cd /kaggle/working/BestNeRF/k-planes

python scripts/prepare_human_annotations_neu3d.py \
  --data-dir data/neu3d/coffee_martini \
  --output-dir /kaggle/working/coffee_martini_human_annotations \
  --split train \
  --cameras auto \
  --max-cameras 1 \
  --frames 0,8,16,24,32,40,48,56 \
  --queries human,hand \
  --downsample 2 \
  --overwrite
```

Use `--cameras auto --split train --max-cameras 3` after the one-camera
workflow is checked. The durable OpenSeg cache contains frames `0..63`, so keep
annotation frame ids inside that range for the current 64-frame checkpoint. The
reviewed final masks should be binary PNGs in `masks/<query>/`.

Generate Grounded-SAM2 proposals:

```bash
cd /kaggle/working

if [ ! -d Grounded-SAM-2 ]; then
  git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
fi

cd /kaggle/working/Grounded-SAM-2
python -m pip install -e .
python -m pip install transformers supervision

mkdir -p checkpoints
if [ ! -f checkpoints/sam2.1_hiera_large.pt ]; then
  wget -O checkpoints/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

cd /kaggle/working/BestNeRF/k-planes

python scripts/propose_human_masks_grounded_sam2.py \
  --annotation-dir /kaggle/working/coffee_martini_human_annotations \
  --grounded-sam2-root /kaggle/working/Grounded-SAM-2 \
  --sam2-config /kaggle/working/Grounded-SAM-2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint /kaggle/working/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --mask-merge union \
  --overwrite
```

For a quick smoke test, add `--limit 2`.

Metrics:

- IoU.
- best IoU over thresholds.
- mIoU.
- precision.
- recall.
- F1.
- AP if keeping heatmaps continuous.

This lets us say:

> On a small manually labeled subset, the model reaches X IoU for query `human`.

This is now the highest-priority next experiment because the masks are reusable
across future checkpoints and do not require preserving every training session.

Evaluate current checkpoint against the reviewed masks:

```bash
cd /kaggle/working/BestNeRF/k-planes

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH

CKPT=$(find /kaggle/input /kaggle/working -path '*cm_semantic_64f_ds16/model.pth' 2>/dev/null | head -1)
echo "$CKPT"

python scripts/evaluate_human_annotations_neu3d.py \
  --annotation-dir /kaggle/working/coffee_martini_human_annotations \
  --config-path plenoxels/configs/local/dynerf_cm_semantic_64f_ds16.py \
  --checkpoint "$CKPT" \
  --teacher-cache-dir /kaggle/input/coffee-martini-openseg-ds16-64f \
  --output-dir /kaggle/working/cm_semantic_64f_ds16_human_eval \
  --batch-size 2048 \
  --fixed-thresholds 0.50,0.75,0.90 \
  --amp
```

This writes:

- `per_annotation_metrics.csv`
- `metrics_by_method_query.csv`
- `metrics_by_method.csv`
- `metrics_by_query.csv`
- `metrics_overall.csv`
- rendered RGB/heatmap/overlay images under `renders/`

With `--teacher-cache-dir`, the semantic rows include both `student_kplanes`
and `teacher_openseg` against the same human-reviewed masks. This is the fairest
first comparison table.

The same run reports PSNR and SSIM for the annotated training frames. For the
final thesis RGB-quality table, still run standard held-out/test-view PSNR,
SSIM, and ideally LPIPS.

### Experiment 4: Semantic Rendering Strategy Ablation

Goal:
Support or reject the claim that semantic volume rendering is better than
single-point feature sampling.

Compare:

- single point at expected depth;
- max density weight sample;
- max semantic-confidence sample;
- top-k weighted samples;
- full weighted rendering.

For top-k, test:

- K = 4
- K = 8
- K = 16
- K = 24
- K = 32
- K = all samples

Report the same metrics for every row:

- AP.
- NDCG.
- AUIoU.
- Pearson.
- Spearman.

This directly answers the critique that K = 24 was not justified.

### Experiment 5: Loss Ablation

Goal:
Find which semantic loss is best and avoid claiming unsupported stability.

Compare:

- cosine loss;
- MSE;
- SmoothL1;
- cosine + SmoothL1;
- normalized-feature MSE.

Keep all other settings fixed:

- same scene;
- same frames;
- same cameras;
- same seed;
- same steps.

Report:

- teacher-imitation metrics;
- RGB PSNR/SSIM/LPIPS;
- training time;
- GPU memory if easy.

### Experiment 6: Targeted Sampling Ablation

Goal:
Prove whether targeted semantic sampling is a real contribution.

Compare:

- uniform random ray sampling;
- teacher high-confidence sampling;
- foreground-biased sampling;
- hard-example sampling where student and teacher disagree;
- mixed uniform + targeted sampling.

At minimum, report:

- final semantic metrics;
- convergence speed;
- whether RGB PSNR changes.

If this experiment is not done, targeted sampling should not be claimed as a
main contribution.

### Experiment 7: Temporal Consistency

Goal:
Show that the 4D semantic field is stable over time.

Simple first metrics:

- frame-to-frame heatmap L1 difference;
- frame-to-frame heatmap Pearson correlation;
- temporal variance inside a manually selected object region.

Stronger later metrics:

- optical-flow warped heatmap consistency;
- object identity consistency;
- track query peak location through time.

Compare:

- semantic model without Temporal-TV;
- semantic model with Temporal-TV;
- teacher OpenSeg heatmaps frame by frame.

### Experiment 8: More Scenes

Goal:
Reduce the "one scene only" criticism.

If time allows, train and evaluate at least one more Neu3D scene.

Good scene criteria:

- visible dynamic foreground;
- object names that OpenSeg can understand;
- different appearance/motion from Coffee Martini.

Even one extra scene improves the thesis claim from "one demo" to "initial
evidence across multiple scenes."

### Experiment 9: RGB Quality Table

Goal:
Support rendering-quality claims.

For each model variant, report:

- PSNR;
- SSIM;
- LPIPS;
- training time;
- checkpoint size;
- render FPS or seconds per frame;
- peak GPU memory if easy.

Rows:

- RGB-only K-Planes baseline.
- RGB + semantic branch from scratch.
- semantic-only after frozen RGB checkpoint.

This checks whether semantic training damages RGB rendering.

### Experiment 10: Define Heatmap-To-Mask Protocol

Goal:
Avoid vague "segmentation" language.

For every query heatmap, define how a binary mask is produced:

- normalize heatmap to [0, 1];
- choose threshold strategy;
- optional connected component filtering;
- optional morphology cleanup.

Threshold choices to compare:

- fixed threshold 0.5;
- top 10 percent pixels;
- Otsu threshold;
- threshold selected on validation masks.

Without this protocol, call the output a "query heatmap", not a "mask" or
"segmentation."

## Suggested Priority Order

1. Treat `human` overlay rendering as done for the current checkpoint.
2. Treat the first `cam01` human-mask evaluation as done.
3. Expand human annotations from one training camera to four training cameras,
   using `human` only.
4. Run thesis-method ablations:
   - cosine/full-weighted baseline;
   - cosine plus normalized SmoothL1;
   - top-k semantic rendering with `K=24`;
   - top-k plus normalized SmoothL1.
5. Add teacher-vs-student heatmap metrics as a secondary table.
6. Defer held-out RGB PSNR/SSIM/LPIPS until the semantic comparison is stable.
7. Run targeted sampling only if we want to keep it as a main claim.
8. Add temporal consistency metrics.
9. Add another scene if time remains.

## Claim Wording Guide

Use this:

> We distill OpenSeg feature maps into a dynamic K-Planes representation and
> show that the student preserves OpenSeg-like open-vocabulary heatmaps on
> Coffee Martini.

Avoid this unless we add human masks and broader evaluation:

> The model accurately segments open-vocabulary objects in 4D dynamic scenes.

Use this:

> The current evaluation measures agreement with the OpenSeg teacher.

Avoid this:

> The current evaluation proves semantic correctness.

Use this:

> Results are demonstrated on Coffee Martini, with broader multi-scene
> validation left as future work.

Avoid this:

> The method generalizes to dynamic 4D scenes.

## Minimum Thesis-Safe Package

If time is short, the minimum package should be:

1. Human query visualizations from our trained model.
2. Model-assisted human annotations for `human`, with every mask reviewed and
   accepted/corrected by a human.
3. Human-mask metrics for the current checkpoint and ablation checkpoints.
4. Teacher-vs-student metrics for 5 to 7 queries.
5. RGB PSNR/SSIM/LPIPS table when time allows.
6. Clear wording that this is OpenSeg distillation, with independent
   human-annotation evaluation added for selected queries rather than a complete
   proof of general semantic segmentation.

That package directly addresses the most serious critique without requiring a
large new benchmark.
