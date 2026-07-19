# Thesis Semantic Branch Reference

This note summarizes the thesis PDF at
`C:\Users\minhc\Downloads\thesis.pdf` and maps it to the current
BestNeRF/K-Planes semantic branch. It is meant as a working reference for future
implementation, experiments, and thesis wording.

## Current Engineering Baseline

The current branch has a working semantic K-Planes path:

- RGB and semantic training run together.
- The semantic branch is separate from the RGB field.
- Semantic geometry inputs and weights can be detached from RGB when
  `semantic_detach_geometry=True`.
- OpenSeg features are loaded from a Kaggle dataset cache:
  `lenguyenminhchau/coffee-martini-openseg-ds16-64f`.
- A full Coffee Martini semantic run completed:
  - config: `cm_semantic_64f_ds16`;
  - 10,000 steps;
  - RGB PSNR around 30;
  - semantic loss around 0.03;
  - checkpoint uploaded as:
    `lenguyenminhchau/coffee-martini-kplanes-semantic-ds16-64f`.
- A render script exists for visual inspection of the `human` query:
  `scripts/render_semantic_query_neu3d.py`.

This is a strong implementation milestone, but it is not yet the complete
method described in the thesis.

## Thesis Core Idea

The thesis proposes distilling OpenSeg features into Dynamic K-Planes so that a
dynamic 4D scene can render both RGB and an open-vocabulary semantic feature
map. At inference time, a text query is encoded with CLIP and compared against
the rendered feature map to produce a query heatmap.

The most thesis-safe short claim is:

> The method distills OpenSeg teacher features into a dynamic K-Planes student
> and preserves teacher-like open-vocabulary query heatmaps on Coffee Martini.

The current evidence should not be phrased as proof of true semantic
segmentation unless we add human-labeled masks.

## Page-Grounded Notes

| Area | Thesis Location | What It Says | Working Interpretation |
| --- | --- | --- | --- |
| Abstract scope | p.14 | No manual semantic masks; evaluation focuses on preserving OpenSeg teacher behavior. | This is the safest scientific framing. Keep it visible. |
| Broad motivation | pp.15-17 | Supports search, segmentation, interaction, and tracking in dynamic scenes. | Good motivation, but the actual evidence is narrower. |
| Contributions | p.17 | Distill OpenSeg into Dynamic K-Planes; compare rendering strategies; propose targeted sampling. | Distillation is implemented. Rendering-strategy and targeted-sampling claims still need direct ablations in our code. |
| Desired output | pp.45-47 | Render RGB plus temporally stable semantic query maps; online text queries produce object localization/segmentation. | Use "query heatmap" unless we define a mask protocol and evaluate masks. |
| OpenSeg cache | pp.48-49 | Thesis describes 16 training cameras, 300 frames, and feature maps around 127x169. | Our current cache is 18 cameras, 64 frames, 126x169, 768 channels, float16. Do not mix these numbers. |
| Semantic decoder | p.50 | Separate semantic K-Planes branch, 4D planes, 768-d OpenSeg output, parallel to RGB. | Current code follows this broad design, with a smaller 64-frame DS16 setup. |
| Top-k semantic rendering | p.51 | Thesis method normalizes per-sample features, aggregates top-k samples by radiance weights, then normalizes again. Main K is 24 of 48 samples. | Current code does full weighted feature rendering, not top-k. |
| Loss | pp.52-53 | Uses cosine plus SmoothL1 on normalized features, with SmoothL1 weight 0.1. | Current code uses cosine semantic loss only; config weight 0.1 scales the semantic objective. |
| Temporal-TV | pp.53-54 | Temporal-TV helps `human` but is not the broad best setting. | Worth trying after the base teacher-metric script is stable. |
| Dataset/evaluation limits | p.55 | Metrics compare student to OpenSeg teacher, not ground-truth semantics. | This should be repeated in any result section. |
| Query set | p.56 | Object, background, and control queries are evaluated with CLIP text embeddings. | Good initial query set for our metric script. |
| Metrics | p.57 | AP, NDCG, AUIoU, Separation, Pearson, Spearman. | Implement these exactly for comparability. |
| Main ablations | pp.58-61 | Volume rendering is better than single-point; cosine+SmoothL1 is best on the broad set. | Supported in thesis tables, but our current branch has not reproduced these ablations yet. |
| Human query | p.61 | Temporal-TV is strongest for `human`. | Useful follow-up if the visual `human` heatmap looks promising. |
| Qualitative caveat | pp.62-63 | Student can look more coherent than OpenSeg, but no manual masks means this is not proof of absolute correctness. | Use this caveat when showing overlays. |
| Failed auxiliary losses | p.64 | Geometry-confidence weighting, affinity distillation, query-ranking, and function-consistent losses perform poorly. | Do not prioritize these unless we need a negative-results section. |
| Conclusion risk | p.67 | Some language claims very high accuracy/rendering quality. | Needs RGB metrics and human-mask metrics before using strong wording. |
| Future work | p.68 | Better teachers, faster architecture, labeled datasets. | Aligns with our next experiment backlog. |

## Thesis Versus Current Code

The thesis final method and our current implementation are close in spirit but
not identical.

### What Matches

- Separate semantic branch parallel to the RGB branch.
- Semantic branch outputs 768-d OpenSeg-like features.
- OpenSeg cache is used as teacher supervision.
- RGB and semantic objectives can train together.
- Semantic loss can be prevented from changing RGB geometry by detaching
  semantic positions, weights, and timestamps.
- Inference compares normalized rendered features to a CLIP text embedding.

### What Does Not Match Yet

- The thesis reports top-k semantic volume rendering. Current code renders
  semantic features with a full weighted sum:
  `sum(weights * semantic_features)`.
- The thesis normalizes semantic features before and after aggregation. Current
  training loss normalizes predicted and target ray features inside the cosine
  loss, but per-sample pre-aggregation normalization is not implemented.
- The thesis uses cosine plus SmoothL1. Current training uses cosine only.
- The thesis discusses K=24 of 48 samples. Current code has no configurable
  semantic top-k.
- The thesis discusses targeted sampling. Current code uses the existing ray
  sampling path.
- The thesis reports broader query metrics. Current verified training only
  confirms loss/PSNR progress; the teacher-vs-student metric script is still a
  next task.
- The thesis describes 16 training cameras and 300 frames. Our current durable
  OpenSeg cache is 18 cameras and 64 frames.

## Claim Safety

Use these claims freely:

- The semantic branch trains end-to-end with RGB on Coffee Martini.
- The model can render a 768-d semantic feature per ray.
- The branch can be queried with CLIP text embeddings to produce heatmaps.
- The semantic checkpoint and OpenSeg cache are persisted as Kaggle datasets.
- Current quantitative semantic training loss shows the student is learning the
  OpenSeg feature target.

Use these only after more evidence:

- "Accurate segmentation": requires human-labeled masks and a thresholding
  protocol.
- "Tracks objects": requires temporal consistency or tracking metrics.
- "General 4D semantics": requires more scenes.
- "High rendering quality": requires PSNR, SSIM, LPIPS, and preferably render
  examples.
- "Top-k is best": requires top-k ablations in our code.
- "Targeted sampling is a contribution": requires a targeted-vs-uniform
  sampling ablation.

## Safe Wording Replacements

| Risky Wording | Safer Wording |
| --- | --- |
| semantic segmentation | open-vocabulary query heatmap |
| semantic accuracy | agreement with OpenSeg teacher |
| ground-truth object mask | teacher-derived high-response region |
| tracks objects through time | renders time-indexed query heatmaps |
| proves open-vocabulary 4D understanding | demonstrates OpenSeg feature distillation in a dynamic K-Planes scene |
| very high rendering quality | RGB PSNR reached about 30 in the Coffee Martini run; add SSIM/LPIPS before stronger claims |

## Best Things To Try Next

### 1. Render Human Query Overlays

Purpose:
Check whether the trained semantic branch is visually meaningful.

Why first:
It is fast, uses the existing checkpoint, and tells us if the project is alive
before building metric machinery.

Status:
Done for the current checkpoint, and the result looked good enough to continue.

Expected output:

- `rgb.png`
- `human_heatmap_gray.png`
- `human_heatmap_color.png`
- `human_overlay.png`
- `human_scores.npy`

Decision:
If the heatmap does not activate on the person/hands at all, pause metrics and
debug feature rendering/checkpoint loading first.

### 2. Build Model-Assisted Human Annotations

Purpose:
Create a reusable evaluation set that does not depend on OpenSeg as both teacher
and judge.

Annotation method:

1. Extract RGB frames from Coffee Martini at the same resolution used for
   evaluation.
2. Use Grounded-SAM2 or an equivalent strong segmentation model to propose
   initial masks from text prompts.
3. Save binary masks and overlay images.
4. Human-review every mask.
5. Correct, reject, or mark ambiguous masks before inclusion.
6. Only accepted or corrected masks enter the benchmark manifest.

Thesis wording:

> We construct a human-annotated evaluation set using a model-assisted
> annotation workflow. Grounded-SAM2 proposes initial masks, and every mask is
> manually reviewed and corrected or rejected before inclusion.

Recommended first queries:

- `human`
- `hand`
- `glass`
- `cup`
- `table`

Recommended first subset:

- cameras: start with `cam00`, then add 1 to 2 more views;
- frames: 8 to 12 frames across the 64-frame semantic cache;
- queries: start with `human` and `hand`, then add object queries.

Suggested layout:

```text
coffee_martini_human_annotations/
  manifest.csv
  images/
    cam00_frame000.png
  masks/
    human/
      cam00_frame000.png
    hand/
      cam00_frame000.png
  overlays/
    human/
      cam00_frame000_overlay.png
```

Suggested manifest columns:

```text
image_path,proposal_path,mask_path,overlay_path,query,camera,frame,status,source,reviewer,notes
```

Valid statuses:

- `accepted`
- `corrected`
- `rejected`
- `ambiguous`

Decision:
This is now the highest-priority dataset artifact. It is reusable across future
checkpoints and lets us report human-mask metrics without saving every model
between sessions.

Preparation command:

```bash
cd /kaggle/working/BestNeRF/k-planes

python scripts/prepare_human_annotations_neu3d.py \
  --data-dir data/neu3d/coffee_martini \
  --output-dir /kaggle/working/coffee_martini_human_annotations \
  --cameras cam00 \
  --frames 0,8,16,24,32,40,48,56 \
  --queries human,hand \
  --downsample 2 \
  --overwrite
```

Notes:

- `--downsample 2` turns the 1014x1352 source videos into 507x676 review
  images, matching the current RGB eval scale.
- Use `--cameras all --max-cameras 3` after the one-camera workflow is checked.
- The script creates `proposals/<query>/`, `masks/<query>/`, and
  `overlays/<query>/`; Grounded-SAM2 should fill proposals/overlays, while the
  reviewed final binary masks go in `masks/<query>/`.

### 3. Compute Human-Mask Metrics

Purpose:
Evaluate the current semantic model against reviewed human annotations.

Metrics:

- IoU at fixed threshold 0.5.
- Best IoU over thresholds.
- AP from continuous heatmaps.
- Precision.
- Recall.
- F1.

Heatmap-to-mask protocol:

- render semantic query score map;
- resize the score map to the annotation mask resolution if needed;
- normalize scores to [0, 1] per image;
- compute AP on continuous scores;
- compute thresholded metrics at fixed and best thresholds.

Decision:
This becomes the main thesis-strength table because the target is no longer
OpenSeg.

### 4. Reproduce Teacher-Vs-Student Metrics

Purpose:
Match the thesis evaluation protocol.

Queries:

- `cup`
- `glass`
- `hand`
- `coffee`
- `martini`
- `table`
- `floor`
- `background`
- `banana`
- `book`
- `car`
- `dog`
- `human`

Metrics:

- AP with teacher top 10 percent as positive.
- NDCG over top 10 percent.
- AUIoU over thresholds 0.50 to 0.95.
- Separation between teacher-high region and the rest.
- Pearson correlation.
- Spearman correlation.

Decision:
This still matters, but it is secondary to human-mask metrics. It shows whether
the student preserved OpenSeg behavior.

### 5. Add Cosine + SmoothL1

Purpose:
Move current training closer to the thesis main method.

Implementation idea:

- Keep cosine loss as the primary semantic objective.
- Add SmoothL1 on L2-normalized predicted and target features.
- Start with `lambda_sl1=0.1`, matching the thesis.
- Keep the existing `semantic_loss_weight` as the outer objective weight unless
  experiments show it needs separation into `semantic_cosine_weight` and
  `semantic_smooth_l1_weight`.

Decision:
Run the same 64-frame setup first. If it improves teacher metrics, scale later.

### 6. Add Semantic Rendering Modes

Purpose:
Test the thesis claim that semantic volume rendering is better than single-point
sampling, and test whether top-k is actually best.

Modes to implement:

- `full_weighted`: current behavior.
- `topk_weighted`: choose top K radiance weights, renormalize, aggregate.
- `max_weight`: use the single sample with highest radiance weight.
- `expected_depth_nearest`: choose sample nearest expected RGB depth.

Top-k values:

- 4
- 8
- 16
- 24
- 32
- all samples

Important detail:
For thesis parity, implement optional per-sample L2 normalization before
aggregation and final L2 normalization after aggregation.

### 5. Add A Small Manual Mask Set

Purpose:
Break the OpenSeg teacher-student evaluation loop.

Minimum useful set:

- 5 to 10 frames.
- 1 to 3 cameras.
- Queries: `human`, `hand`, maybe `glass`.

Metrics:

- IoU.
- Precision.
- Recall.
- F1.
- AP from continuous heatmaps.

Decision:
Even a tiny manual set turns the result from "teacher imitation only" into an
actual semantic validation section.

### 6. RGB Quality Evaluation

Purpose:
Support claims that semantic training does not damage reconstruction.

Rows:

- RGB-only K-Planes.
- RGB + semantic branch from scratch.
- Semantic-only branch from frozen RGB checkpoint, if we run it.

Metrics:

- PSNR.
- SSIM.
- LPIPS.
- training time.
- checkpoint size.
- render time per frame.

Decision:
Do this before using "high rendering quality" wording.

### 7. Temporal Consistency

Purpose:
Support dynamic-scene claims.

Simple first metrics:

- frame-to-frame heatmap L1 difference;
- frame-to-frame heatmap Pearson correlation;
- query peak motion smoothness.

Stronger later metric:

- optical-flow-warped heatmap consistency.

Decision:
Try this especially for `human`, because the thesis suggests Temporal-TV helps
that query.

### 8. Temporal-TV Variant

Purpose:
Test the thesis observation that Temporal-TV helps dynamic foreground objects.

Suggested order:

1. Implement basic temporal TV over semantic time planes.
2. Train the same 64-frame Coffee Martini setup.
3. Compare only against the current branch first.
4. Then compare against SmoothL1/top-k once those exist.

Decision:
Promote Temporal-TV only if it improves `human` without hurting the broad query
set too much.

### 9. Targeted Sampling Ablation

Purpose:
Support or remove the targeted-sampling contribution.

Variants:

- uniform ray sampling;
- teacher high-confidence sampling;
- foreground-biased sampling;
- hard-example sampling where teacher/student disagree;
- mixed uniform plus targeted sampling.

Decision:
If we cannot run this, keep targeted sampling as future work rather than a main
contribution.

### 10. Another Scene

Purpose:
Reduce the "one scene" criticism.

Good scene criteria:

- clear dynamic foreground;
- object names CLIP/OpenSeg can understand;
- manageable cache size;
- different appearance from Coffee Martini.

Decision:
One extra scene is enough to make the claim less brittle. More than that is
nice but probably expensive.

## Recommended Roadmap

### Stage A: Verify Current Checkpoint

1. Reconnect Kaggle.
2. Restore datasets:
   - `lenguyenminhchau/coffee-martini-openseg-ds16-64f`;
   - `lenguyenminhchau/coffee-martini-kplanes-semantic-ds16-64f`.
3. Render `human` overlays.
4. Save overlays as a small Kaggle output/dataset.

Exit condition:
At least one rendered frame clearly shows a plausible `human` response.

Status:
Done for the current checkpoint.

### Stage B: Build The Human Annotation Set

1. Extract eval RGB frames from selected cameras/timestamps.
2. Generate initial masks with Grounded-SAM2.
3. Save masks and overlays.
4. Human-review every mask.
5. Keep only accepted/corrected masks in `manifest.csv`.
6. Save the annotation set as a durable dataset.

Exit condition:
A reviewed human annotation set exists for at least `human` and `hand`.

### Stage C: Build The Human-Metric Script

1. Render student query heatmaps for every annotation row.
2. Resize/normalize heatmaps to match mask resolution.
3. Compute AP, IoU, best IoU, precision, recall, and F1.
4. Write CSV/JSON tables.

Exit condition:
A human-mask metric table exists for the current cosine-only/full-weighted
branch.

### Stage D: Build The Teacher-Metric Script

1. Render student semantic features for chosen camera/frame/query set.
2. Load OpenSeg cached teacher features for matching frames.
3. Encode all text queries with the same CLIP model.
4. Compute the thesis metrics.
5. Write CSV/JSON tables.

Exit condition:
A table exists for the current cosine-only/full-weighted branch.

### Stage E: Match The Thesis Method

1. Add normalized SmoothL1.
2. Add semantic rendering modes.
3. Run `full_weighted` vs `topk_weighted K=24`.
4. Compare against both human-mask and teacher-agreement metrics.

Exit condition:
We know whether the thesis final recipe improves our current code.

### Stage F: Strengthen The Science

1. Add RGB PSNR/SSIM/LPIPS.
2. Add Temporal-TV only if the `human` query remains important.
3. Add another scene only if time and Kaggle storage allow.

Exit condition:
We can defend the thesis claims with human annotations, teacher agreement, and
RGB quality evidence.

## Practical Notes For Kaggle

- The OpenSeg feature cache is large. The current durable DS16/64F cache is
  about 36 GB.
- Avoid copying large `.npy` files inside `/kaggle/working`; use symlinks or
  upload from a temp location when possible.
- Quick Save does not stop the running notebook.
- `/kaggle/temp` does not survive restart.
- `/kaggle/working` may appear persistent during a session, but treat Kaggle
  datasets and GitHub as the durable storage.
- If SSH host verification fails after restart, remove the stale ngrok
  `known_hosts` entry for the new port.
- If `torch.cuda.is_available()` is false but `/dev/nvidia*` exists, export:

```bash
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
```

## Bottom Line

The thesis already contains the most important caveat: without human masks,
evaluation measures OpenSeg teacher preservation, not absolute semantic
correctness. P0 visual inspection is done, so the next best move is not another
large training run. It is:

1. build model-assisted human annotations and review every mask;
2. compute human-mask metrics for the current checkpoint;
3. reproduce teacher-vs-student metrics as a secondary table;
4. add SmoothL1/top-k to match the thesis method and compare again.
