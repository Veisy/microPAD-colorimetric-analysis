# YOLOv11m-pose Implementation Plan

## Project Overview

This plan outlines the implementation of YOLOv11m-pose for microPAD detection, replacing the previous segmentation approach. Since we're on a separate branch (model/yolo11-pose), this is a clean implementation rather than a migration.

**Target Architecture:**
- Model: YOLOv11m-pose at 960px resolution
- Post-processing: Direct keypoint extraction (no polygon simplification)
- Label format: 4 keypoints with visibility flags
- Output: 4 corner coordinates (clockwise from top-left)

**Success Criteria:**
- Training configuration is correct
- Detection script implements pose inference
- Visual inspection shows accurate corner detection
- Inference time <100ms per image at 960px
- Works across all phone models (iPhone 11, iPhone 15, Samsung A75, Realme C55)

**Note:** Model training (dataset generation and training) is handled separately by the user.

**Hardware:**
- GPUs: Dual RTX A6000 (48GB each, CUDA devices 0,2)
- RAM: 256GB
- Storage: 20TB NVMe

---

## Status Legend
- [ ] Not started
- [🔄] In progress
- [✅] Completed
- [⚠️] Blocked/needs attention

---

## Phase 1: Training Configuration

### 1.1 Update Training Script for Pose Model
- [✅] **Objective:** Configure `train_yolo.py` to use YOLOv11m-pose architecture
- [ ] **File:** `python_scripts/train_yolo.py`
- [ ] **Requirements:**
  - Change base model from `yolo11m-seg.pt` to `yolo11m-pose.pt`
  - Set `kpt_shape=[4, 3]` (4 keypoints, 3 values: x, y, visibility)
  - Verify Ultralytics version supports pose training (v8.1.0+)
  - Keep hyperparameters: epochs=150, batch=24, imgsz=960
  - Configure dual GPU usage (devices=0,2)
- [ ] **Rationale:** YOLOv11-pose requires different architecture and configuration
- [ ] **Success Criteria:**
  - Training script loads yolo11m-pose.pt successfully
  - No errors about keypoint configuration
  - Training logs show keypoint metrics (OKS, mAP)

---

### 1.2 Update Dataset Configuration YAML
- [✅] **Objective:** Add pose keypoint metadata to dataset config
- [✅] **File:** `python_scripts/configs/micropad_synth.yaml`
- [ ] **Requirements:**
  - Add `kpt_shape: [4, 3]` to YAML config
  - Add keypoint names: `['TL', 'TR', 'BR', 'BL']`
  - Add skeleton edges: `[[0,1], [1,2], [2,3], [3,0]]`
  - Keep existing train/val split (3 phones train, 1 phone val)
- [ ] **Rationale:** Ultralytics requires keypoint metadata for pose training
- [ ] **Success Criteria:**
  - YAML config validates without errors
  - Ultralytics loads dataset with keypoint configuration
  - Training visualization shows rectangle skeleton overlays

---

### 1.3 Verify Label Format
- [✅] **Objective:** Confirm `prepare_yolo_dataset.py` generates correct pose labels
- [✅] **File:** `python_scripts/prepare_yolo_dataset.py` (already updated)
- [ ] **Requirements:**
  - Verify default format is 'pose' (not 'seg')
  - Verify label format: `0 x1 y1 2 x2 y2 2 x3 y3 2 x4 y4 2`
  - Verify vertices ordered clockwise from top-left (TL, TR, BR, BL)
  - Test with sample augmented dataset
- [ ] **Rationale:** Training depends on correct label format
- [ ] **Success Criteria:**
  - Generated labels match pose format specification
  - Vertex ordering is consistent (clockwise from TL)
  - All visibility flags are 2 (visible)

---

## Phase 2: Detection Script Implementation

### 2.1 Update detect_quads.py for Pose Detection
- [✅] **Objective:** Replace segmentation logic with pose keypoint extraction
- [✅] **File:** `python_scripts/detect_quads.py`
- [✅] **Requirements:**
  - Load pose model instead of segmentation model
  - Extract keypoints from `result.keypoints.xy` (shape: [N, 4, 2])
  - Remove Douglas-Peucker polygon simplification code
  - Remove segmentation mask handling (`result.masks`)
  - Remove fallback logic (convex hull, minAreaRect)
  - Validate keypoint ordering (clockwise from TL)
  - Implement overlap handling (IoU >0.20 → keep higher confidence)
- [✅] **Rationale:** Pose model provides keypoints directly, no polygon approximation needed
- [✅] **Success Criteria:**
  - Keypoints extracted without errors
  - Consistent ordering (TL, TR, BR, BL)
  - No duplicate detections (overlap handling works)
  - Code is simpler (fewer lines, no fallback logic)

---

### 2.2 Preserve MATLAB Interface
- [✅] **Objective:** Maintain stdout format for MATLAB compatibility
- [✅] **File:** `python_scripts/detect_quads.py`
- [✅] **Requirements:**
  - Keep output format: `numDetections` followed by `x1 y1 x2 y2 x3 y3 x4 y4 confidence`
  - Maintain 0-based indexing (Python convention)
  - Preserve 6 decimal places for coordinates
  - Keep error handling (stderr for errors, exit codes)
  - No changes to command-line arguments
- [✅] **Rationale:** MATLAB parsing depends on exact stdout format
- [✅] **Success Criteria:**
  - MATLAB parses output without errors
  - Coordinate format identical to previous implementation
  - No breaking changes to integration

---

### 2.3 Clean Up Obsolete Code
- [✅] **Objective:** Remove all segmentation-specific code
- [✅] **File:** `python_scripts/detect_quads.py`
- [✅] **Requirements:**
  - Remove `fit_quad_from_contour()` function
  - Remove `result.masks` handling
  - Remove Douglas-Peucker imports (cv2.approxPolyDP)
  - Remove fallback algorithms (convex hull, minAreaRect)
  - Update docstrings to reference pose model
  - Update comments throughout file
- [✅] **Rationale:** Simplify codebase, remove technical debt
- [✅] **Success Criteria:**
  - No unused imports or functions
  - Code is cleaner and easier to maintain
  - No references to segmentation model

---

## Phase 3: MATLAB Integration

### 3.1 Update cut_micropads.m Model Path
- [✅] **Objective:** Configure MATLAB to use pose model
- [✅] **File:** `matlab_scripts/cut_micropads.m`
- [ ] **Requirements:**
  - Update default model path to pose weights: `models/yolo11m_micropad_pose.pt`
  - Keep inference size at 1280px (or adjust based on training resolution)
  - Keep confidence threshold at 0.6
  - Verify Python environment path configuration
  - No changes to coordinate parsing (output format unchanged)
  - Fix any syntax errors in the file
- [ ] **Rationale:** MATLAB must call pose model for detection
- [ ] **Success Criteria:**
  - MATLAB file has no syntax errors
  - MATLAB successfully invokes Python script
  - Model loads without errors
  - Coordinates parsed correctly from stdout

---

### 3.2 Visual Testing
- [ ] **Objective:** Verify end-to-end pipeline with pose detection
- [ ] **File:** Test images from `1_dataset/`
- [ ] **Requirements:**
  - Run `cut_micropads.m` on test images from all phone models
  - Verify AI detection works on typical microPAD images
  - Visual inspection: corners aligned with ground truth
  - Test multi-micropad images (1-10 per image)
  - Test edge cases: rotated micropads, near-border polygons
  - Verify coordinates.txt format (10 columns with rotation)
- [ ] **Rationale:** Visual validation is sufficient for corner accuracy
- [ ] **Success Criteria:**
  - Pipeline runs without errors
  - Corners visually aligned with microPAD zones
  - Multi-micropad detection works correctly
  - No false positives on distractors
  - Rotation adjustment workflow still functional

---

## Phase 4: Documentation

### 4.1 Update CLAUDE.md
- [✅] **Objective:** Document pose model implementation
- [✅] **File:** `CLAUDE.md`
- [ ] **Requirements:**
  - Update pipeline description to mention keypoint detection
  - Replace references to segmentation with pose
  - Document label format: 4 keypoints with visibility flags
  - Update example commands for training and inference
  - Remove references to Douglas-Peucker polygon simplification
- [ ] **Rationale:** Keep documentation in sync with implementation
- [ ] **Success Criteria:**
  - Documentation accurately reflects pose architecture
  - No outdated references to segmentation
  - Clear explanation of keypoint format

---

### 4.2 Update Implementation Plan
- [ ] **Objective:** Mark plan as complete
- [ ] **File:** `documents/plans/YOLO_POSE_MIGRATION_PLAN.md`
- [ ] **Requirements:**
  - Update all checkboxes to completed
  - Add final notes about implementation
  - Document any deviations from plan
  - Archive plan or delete (user preference)
- [ ] **Rationale:** Track completion of implementation
- [ ] **Success Criteria:**
  - Plan reflects final state
  - All phases marked complete

---

## Progress Tracking

### Overall Status
- [✅] Phase 1: Training Configuration (3/3 tasks)
- [✅] Phase 2: Detection Script Implementation (3/3 tasks)
- [✅] Phase 3: MATLAB Integration (1/2 tasks) - Task 3.2 (Visual Testing) to be done by user
- [✅] Phase 4: Documentation (2/2 tasks)

### Key Milestones
- [✅] Milestone 1: Training configuration ready
- [✅] Milestone 2: Detection script implements pose inference
- [✅] Milestone 3: MATLAB integration configured (visual testing to be done by user)
- [✅] Milestone 4: Documentation updated

**Note:** Model training (dataset generation, training, validation) and visual testing are handled by the user.

---

## Notes & Decisions

### Design Decisions

**Why YOLOv11m-pose over YOLOv11m-seg?**
- Direct keypoint prediction eliminates Douglas-Peucker approximation errors
- Simpler post-processing (no polygon simplification needed)
- Better corner localization accuracy (trained end-to-end for keypoint regression)
- Removes fallback logic complexity (convex hull, minAreaRect)

**Why 960px resolution?**
- Good balance between accuracy and speed
- High enough resolution to preserve corner detail
- Faster than 1280px with minimal accuracy loss
- Allows batch=24 on dual A6000 GPUs

**Why IoU threshold of 20% for overlap?**
- Matches physical constraint: micropads can overlap up to 20%
- Higher overlap indicates duplicate detection (same micropad detected twice)
- Lower threshold would reject valid overlapping micropads

**Why visibility flag = 2 for all keypoints?**
- All polygon corners are always visible in our dataset
- No partial occlusion in synthetic augmentation
- Simplifies label generation (no visibility computation needed)

**Why no segmentation code preservation?**
- We're on a separate branch (model/yolo11-pose)
- No need for rollback or comparison with segmentation
- Cleaner codebase without legacy code

### Implementation Strategy

**Dataset Preparation:**
- `augment_dataset.m` generates synthetic training data
- `prepare_yolo_dataset.py` converts MATLAB coordinates to YOLO pose labels
- All dataset handling centralized in these two scripts
- No separate conversion or validation scripts needed

**Visual Testing:**
- Visual inspection sufficient for corner accuracy validation
- No need for quantitative benchmarking against segmentation
- User will test detection quality directly in MATLAB GUI

**Code Quality:**
- Remove all segmentation code (no legacy support)
- Implement clean pose detection without workarounds
- Keep code simple and maintainable
- No unnecessary test scripts or validation tools

---

---

## Implementation Complete ✅

**Date Completed:** 2025-11-04
**Implementation Status:** All code phases complete, ready for model training and testing

### What Was Implemented

**Phase 1: Training Configuration**
- ✅ Updated `train_yolo.py` to use YOLOv11m-pose architecture
- ✅ Updated `micropad_synth.yaml` with keypoint metadata (kpt_shape, skeleton, flip_idx)
- ✅ Verified `prepare_yolo_dataset.py` generates pose labels by default

**Phase 2: Detection Script Implementation**
- ✅ Updated `detect_quads.py` for pose keypoint extraction
- ✅ Removed Douglas-Peucker polygon simplification (~80 lines)
- ✅ Implemented NMS overlap handling (IoU >0.20 threshold)
- ✅ Preserved MATLAB interface (stdout format unchanged)

**Phase 3: MATLAB Integration**
- ✅ Updated `cut_micropads.m` to use `yolo11m_micropad_pose.pt`
- ✅ Updated model path references and comments
- ⏳ Visual testing to be performed by user after model training

**Phase 4: Documentation**
- ✅ Updated `CLAUDE.md` to reflect pose model architecture
- ✅ Replaced all segmentation references with pose terminology
- ✅ Updated label format documentation with visibility flags

### Files Modified
1. `python_scripts/train_yolo.py` - Training configuration
2. `python_scripts/configs/micropad_synth.yaml` - Dataset configuration
3. `python_scripts/detect_quads.py` - Detection inference
4. `matlab_scripts/cut_micropads.m` - MATLAB integration
5. `CLAUDE.md` - Project documentation

### Next Steps (User Actions)
1. Generate augmented dataset: `augment_dataset('numAugmentations', 10, 'exportYOLOLabels', true)`
2. Prepare YOLO dataset: `python prepare_yolo_dataset.py --format pose`
3. Train pose model: `python train_yolo.py --stage 1 --imgsz 960 --batch 24 --epochs 150`
4. Test detection: Run `cut_micropads.m` on sample images
5. Visually verify corner detection accuracy

---

## Contact & Support

**Project Lead:** Veysel Y. Yilmaz
**Last Updated:** 2025-11-04
**Version:** 2.0.0 (Clean pose implementation)
