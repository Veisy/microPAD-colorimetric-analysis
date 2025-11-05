## Phase 1 — Training Script Hygiene
- [ ] Replace hard-coded GPU selection in `python_scripts/train_yolo.py` with a configurable constant sourced from CLI args or environment (eliminate `'0,2'` magic number).
- [ ] Rename stage identifiers and experiment names to reflect the YOLOv11m pose workflow (e.g., `yolo11m_pose_stage2`) for clarity in logging and checkpoints.
- [ ] Introduce named constants (or argparse defaults) for key hyperparameters currently inlined (e.g., stage-2 learning rate) to remove residual magic numbers.
- **Verification**: `python train_yolo.py --help` shows updated defaults; dry-run `python python_scripts/train_yolo.py --stage 1 --epochs 1` completes without device parsing warnings.

## Phase 2 — CLI Flexibility and Best Practices
- [ ] Expose CLI flags for loader workers, caching strategy, SyncBatchNorm, optimizer, and learning-rate schedule so workstation runs need no code edits.
- [ ] Fix export precision flag logic (pair `--half` / `--no-half`) to allow explicit FP32 export.
- [ ] Ensure all user-facing descriptions avoid workstation-specific references.
- **Verification**: CLI accepts new arguments without error; `python train_yolo.py --export --weights dummy.pt --no-half` prints FP32 intent.

## Phase 3 — Workstation-Oriented Defaults (Applied on target machine)
- [ ] Document recommended launch commands (batch sizes, patience, epochs) in README or accompanying ops doc, using symbolic placeholders instead of workstation identifiers.
- [ ] Outline dataset preparation workflow emphasizing rerun of `prepare_yolo_dataset.py` post-migration (no absolute paths baked into repo files).
- [ ] Capture validation/monitoring checklist (metrics thresholds, sync-bn enablement) in docs without hardware-specific constants.
- **Verification**: Team walkthrough confirms docs are host-agnostic; dry run on workstation follows plan without edits to source beyond repo defaults.
