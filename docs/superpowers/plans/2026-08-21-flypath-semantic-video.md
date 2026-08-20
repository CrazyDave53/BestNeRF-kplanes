# Flypath Semantic Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Kaggle-friendly renderer that turns a saved Neu3D fly path into semantic heatmap videos and per-frame similarity maps.

**Architecture:** Add one standalone script that reuses the existing query sample renderer helpers for config loading, trainer construction, CLIP text encoding, score normalization, heatmap coloring, and PNG writing. The script renders arbitrary `c2w` poses from the fly-path JSON while driving scene time with either a fixed frame or a 0..63..0 ping-pong schedule.

**Tech Stack:** Python, PyTorch, existing K-Planes trainer/model code, OpenCLIP, NumPy, PIL, ffmpeg.

---

### Task 1: Add Flypath Video Renderer

**Files:**
- Create: `scripts/render_flypath_semantic_video_neu3d.py`

- [x] **Step 1: Implement CLI and fly-path loading**

The script accepts `--flypath`, `--config-path`, `--checkpoint`, `--queries`, `--output-dir`, `--time-mode`, `--num-time-frames`, `--fixed-time-frame`, `--batch-size`, `--cmap`, `--alpha`, `--fps`, `--max-frames`, and config overrides.

- [x] **Step 2: Render arbitrary fly-path poses**

For each fly-path pose, build rays from the dataset intrinsics and JSON `c2w`, use NDC ray conversion from existing utilities, feed the semantic model in no-grad training mode, and compute cosine similarity against text embeddings.

- [x] **Step 3: Save outputs**

Write common RGB frames, per-query heatmaps, per-query overlays, and raw score maps. Write `metadata.csv` mapping render index to source fly-path index and raw time frame.

- [x] **Step 4: Build videos**

Use ffmpeg to create forward MP4s from overlay frames and loop MP4s from a concat list that appends the reverse frame order.

- [x] **Step 5: Verify**

Run `python -m py_compile scripts/render_flypath_semantic_video_neu3d.py scripts/render_query_samples_neu3d.py scripts/make_neu3d_flypath.py`.

