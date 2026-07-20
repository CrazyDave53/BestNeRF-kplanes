# Coarse-To-Fine Semantic Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Kaggle-ready coarse-to-fine semantic training scripts for the strongest Top-K semantic variants.

**Architecture:** Use a two-stage training flow. Stage 1 trains a coarse semantic field with `semantic_multiscale_res=[1]`; Stage 2 resumes from Stage 1, expands to `semantic_multiscale_res=[1, 2]`, skips optimizer/scheduler state, and trains to the same final step budget as previous ablations.

**Tech Stack:** Python config files, Bash queue scripts, `unittest`.

---

### Task 1: Coarse/Fine Configs

**Files:**
- Create: `plenoxels/configs/local/dynerf_cm_semantic_64f_ds16_c2f_coarse.py`
- Create: `plenoxels/configs/local/dynerf_cm_semantic_64f_ds16_c2f_fine.py`
- Test: `tests/test_semantic_c2f_queue.py`

- [ ] Write failing tests proving env-driven expnames, render modes, Top-K values, SmoothL1 weights, stage step counts, semantic scales, and skipped optimizer/scheduler loading.
- [ ] Implement the coarse and fine config files by extending `dynerf_cm_semantic_64f_ds16.py`.
- [ ] Run `python -m unittest tests.test_semantic_c2f_queue -v`.

### Task 2: Smoke And Full Queue Scripts

**Files:**
- Create: `scripts/kaggle_smoke_semantic_c2f_queue.sh`
- Create: `scripts/kaggle_run_semantic_c2f_queue.sh`
- Test: `tests/test_semantic_c2f_queue.py`

- [ ] Extend tests to verify the smoke queue and full queue include `topk8`, `topk16`, `topk24`, and `topk16_smooth010`.
- [ ] Implement smoke script with 20-step coarse and 40-step fine stages.
- [ ] Implement full queue script with 3000-step coarse and 10000-step fine stages, final checkpoint packaging, and optional Kaggle dataset upload.
- [ ] Run `python -m unittest tests.test_semantic_c2f_queue -v`.

### Task 3: Docs And Verification

**Files:**
- Modify: `docs/kaggle_runbook.md`
- Modify: `docs/semantic_evaluation_risks_and_next_steps.md`

- [ ] Add the C2F smoke/full commands and the intended finalist variants.
- [ ] Run focused semantic tests:
  `python -m unittest tests.test_semantic_c2f_queue tests.test_semantic_ablation_queue -v`
- [ ] Run `git diff --check`.
