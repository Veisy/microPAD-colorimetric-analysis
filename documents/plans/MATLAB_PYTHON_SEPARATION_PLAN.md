# MATLAB-Python Pipeline Separation Plan

## Project Overview

Refactor the microPAD colorimetric analysis codebase to establish clear separation of concerns between MATLAB and Python:

**Problem:** Current implementation has MATLAB scripts handling AI model training concerns (YOLO label export in `augment_dataset.m`), which tightly couples data processing with AI training pipeline. This violates single-responsibility principle and makes the architecture harder to maintain.

**Target Deliverables:**
- MATLAB scripts focused exclusively on data processing (1→2→3 pipeline)
- Python scripts handling all AI model training and inference operations
- Clean interface between MATLAB and Python for AI predictions (via Python helper script)
- Updated documentation reflecting new architecture

**Success Criteria:**
- `augment_dataset.m` follows same 1→2→3 structure as original pipeline (no YOLO-specific code)
- Python scripts handle all YOLO label generation (read MATLAB coordinates, create labels)
- `cut_micropads.m` communicates with Python via dedicated inference helper script
- All documentation updated to reflect separation of concerns
- No functionality lost in refactoring

---

## Status Legend
- [ ] Not started
- [🔄] In progress
- [✅] Completed
- [⚠️] Blocked/needs attention
- [🔍] Needs review

---

## Phase 1: Refactor augment_dataset.m (Remove AI Training Concerns)

### 1.1 Remove YOLO Label Export Functionality
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\matlab_scripts\augment_dataset.m`
- [ ] **Task:** Remove `exportYOLOLabels` parameter and all YOLO label export code
- [ ] **Changes:**
  - Line 47: Delete `exportYOLOLabels` parameter documentation
  - Line 151: Delete `addParameter(parser, 'exportYOLOLabels', true, @islogical);`
  - Line 205: Delete `cfg.exportYOLOLabels = opts.exportYOLOLabels;`
  - Lines 541-543: Delete YOLO label export call in passthrough section
  - Lines 908-911: Delete YOLO label export call in augmentation section
  - Lines 3122-3163: Delete entire `export_yolo_segmentation_labels()` function and `order_corners_clockwise()` helper
- [ ] **Rationale:** YOLO label generation is AI training concern, should be handled by Python scripts. MATLAB should only focus on creating augmented image pipeline (1→2→3 structure).
- [ ] **Test:** Run `augment_dataset('numAugmentations', 1)` and verify:
  - No `labels/` subdirectories created in `augmented_1_dataset/[phone]/`
  - All other outputs (images, coordinates) remain unchanged
  - No errors or warnings about missing exportYOLOLabels parameter

---

### 1.2 Update augment_dataset.m Documentation
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\matlab_scripts\augment_dataset.m`
- [ ] **Task:** Update header comments to reflect removal of YOLO export
- [ ] **Changes:**
  - Lines 12-14: Remove references to "YOLO label export" from FEATURES section
  - Lines 33-35: Update OUTPUT STRUCTURE comment to only mention stages 1-3:
    ```matlab
    % OUTPUT STRUCTURE:
    %   augmented_1_dataset/[phone]/           - Real copies + synthetic scenes
    %   augmented_2_micropads/[phone]/con_*/   - Polygon crops + coordinates.txt
    %   augmented_3_elliptical_regions/[phone]/con_*/ - Elliptical patches + coordinates.txt
    ```
  - Lines 47-48: Delete example showing `exportYOLOLabels` parameter
- [ ] **Rationale:** Documentation should accurately reflect current functionality after YOLO export removal
- [ ] **Test:** Read header comments and verify no references to YOLO labels remain

---

### 1.3 Verify augment_dataset.m Pipeline Structure
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\matlab_scripts\augment_dataset.m`
- [ ] **Task:** Ensure augment_dataset.m follows standard 1→2→3 pipeline structure (same as original pipeline)
- [ ] **Verification Points:**
  - Input: `1_dataset/`, `2_micropads/coordinates.txt`, `3_elliptical_regions/coordinates.txt`
  - Output: `augmented_1_dataset/`, `augmented_2_micropads/`, `augmented_3_elliptical_regions/`
  - No AI model training artifacts (labels, annotations) generated
  - Coordinates stored in standard format (10-column for stage 2, 7-column for stage 3)
- [ ] **Rationale:** MATLAB augmentation should mirror original pipeline structure for consistency
- [ ] **Test:** Run full augmentation and verify output directory structure matches expectations

---

## Phase 2: Enhance Python Scripts (Take Over AI Training Responsibilities)

### 2.1 Add YOLO Label Generation to prepare_yolo_dataset.py
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\python_scripts\prepare_yolo_dataset.py`
- [ ] **Task:** Add function to read MATLAB coordinates and generate YOLO segmentation labels
- [ ] **New Function (add after line 76):**
  ```python
  def generate_yolo_labels() -> Tuple[int, int]:
      """Generate YOLO segmentation labels from MATLAB coordinates.

      Reads polygon coordinates from augmented_2_micropads/[phone]/coordinates.txt
      and creates YOLOv11 segmentation labels in augmented_1_dataset/[phone]/labels/.

      Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized [0,1])

      Returns:
          Tuple of (total_labels_created, total_polygons_processed)
      """
      import numpy as np
      from pathlib import Path

      total_labels = 0
      total_polygons = 0

      micropads_dir = PROJECT_ROOT / "augmented_2_micropads"

      for phone in PHONE_DIRS:
          coord_file = micropads_dir / phone / "coordinates.txt"
          if not coord_file.exists():
              print(f"⚠️  Warning: coordinates.txt not found for {phone}")
              continue

          labels_dir = AUGMENTED_DATASET / phone / "labels"
          labels_dir.mkdir(parents=True, exist_ok=True)

          # Read coordinates (format: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation)
          with open(coord_file, 'r') as f:
              lines = f.readlines()

          # Group polygons by image
          image_polygons = {}
          for line in lines:
              parts = line.strip().split()
              if len(parts) < 10:
                  continue

              img_name = parts[0]
              # Remove _con_N suffix to get base image name
              base_name = '_'.join(img_name.split('_')[:-2])

              # Extract polygon vertices (columns 2-9: x1 y1 x2 y2 x3 y3 x4 y4)
              vertices = [float(parts[i]) for i in range(2, 10)]
              polygon = [(vertices[i], vertices[i+1]) for i in range(0, 8, 2)]

              if base_name not in image_polygons:
                  image_polygons[base_name] = []
              image_polygons[base_name].append(polygon)

          # Get image dimensions and create labels
          images_dir = AUGMENTED_DATASET / phone / "images"
          for base_name, polygons in image_polygons.items():
              # Find corresponding image file
              img_path = None
              for ext in ['.jpg', '.jpeg', '.png']:
                  candidate = images_dir / f"{base_name}{ext}"
                  if candidate.exists():
                      img_path = candidate
                      break

              if img_path is None:
                  print(f"⚠️  Warning: image not found for {base_name}")
                  continue

              # Get image dimensions
              import cv2
              img = cv2.imread(str(img_path))
              if img is None:
                  continue
              height, width = img.shape[:2]

              # Write label file
              label_path = labels_dir / f"{base_name}.txt"
              with open(label_path, 'w') as f:
                  for polygon in polygons:
                      # Normalize coordinates to [0, 1]
                      norm_poly = [(x / width, y / height) for x, y in polygon]
                      # Write: 0 x1 y1 x2 y2 x3 y3 x4 y4
                      coords_str = ' '.join([f"{x:.6f} {y:.6f}" for x, y in norm_poly])
                      f.write(f"0 {coords_str}\n")
                      total_polygons += 1

              total_labels += 1

      print(f"✅ Generated {total_labels} label files ({total_polygons} polygons)")
      return total_labels, total_polygons
  ```
- [ ] **Rationale:** Python handles all AI training data preparation. Reading MATLAB coordinates ensures single source of truth (MATLAB generates coordinates, Python converts to AI format).
- [ ] **Test:** Run function and verify:
  - Label files created in `augmented_1_dataset/[phone]/labels/`
  - Label format matches YOLOv11 segmentation (class_id + 4 normalized vertex pairs)
  - All polygons from coordinates.txt are represented

---

### 2.2 Integrate Label Generation into prepare_yolo_dataset.py Workflow
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\python_scripts\prepare_yolo_dataset.py`
- [ ] **Task:** Call `generate_yolo_labels()` in main workflow
- [ ] **Changes:**
  - Line 255: Add call after `verify_labels()` check:
    ```python
    # Generate YOLO labels from MATLAB coordinates
    print("Generating YOLO labels from MATLAB coordinates...")
    label_count, polygon_count = generate_yolo_labels()
    print()
    ```
  - Update `print_summary()` to include label generation statistics (add parameters)
- [ ] **Rationale:** Automated label generation ensures YOLO training data is always synchronized with MATLAB coordinates
- [ ] **Test:** Run `python prepare_yolo_dataset.py` and verify:
  - Labels generated before train/val split creation
  - Summary shows label and polygon counts
  - No missing label warnings after generation

---

### 2.3 Add Import Statements for New Dependencies
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\python_scripts\prepare_yolo_dataset.py`
- [ ] **Task:** Add required imports for label generation
- [ ] **Changes (after line 24):**
  ```python
  import numpy as np
  import cv2
  ```
- [ ] **Rationale:** `numpy` for coordinate manipulation, `cv2` for reading image dimensions
- [ ] **Test:** Run script and verify no import errors

---

## Phase 3: Create Python Inference Helper Script

### 3.1 Create detect_quads.py Inference Script
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\python_scripts\detect_quads.py` (NEW)
- [ ] **Task:** Create standalone Python script for YOLO inference callable from MATLAB
- [ ] **Implementation:**
  ```python
  """
  Standalone YOLO inference script for microPAD quad detection.

  Called by MATLAB cut_micropads.m to perform AI-based polygon detection.
  Accepts image path and outputs detected quad coordinates to stdout.

  Usage:
      python detect_quads.py <image_path> <model_path> [--conf THRESHOLD] [--imgsz SIZE]

  Output Format (stdout):
      numDetections
      x1 y1 x2 y2 x3 y3 x4 y4 confidence
      x1 y1 x2 y2 x3 y3 x4 y4 confidence
      ...
  """

  import sys
  import argparse
  from pathlib import Path
  import numpy as np
  import cv2


  def parse_args():
      parser = argparse.ArgumentParser(
          description="YOLO quad detection for microPAD analysis"
      )
      parser.add_argument("image_path", type=str, help="Path to input image")
      parser.add_argument("model_path", type=str, help="Path to YOLO model (.pt)")
      parser.add_argument(
          "--conf", type=float, default=0.6, help="Confidence threshold (default: 0.6)"
      )
      parser.add_argument(
          "--imgsz", type=int, default=640, help="Inference image size (default: 640)"
      )
      return parser.parse_args()


  def fit_quad_from_contour(contour):
      """Fit quadrilateral from YOLO contour points using Douglas-Peucker."""
      if len(contour) < 4:
          return None

      # Calculate perimeter
      perimeter = cv2.arcLength(contour, closed=True)

      # Try Douglas-Peucker with increasing epsilon
      for epsilon_factor in [0.02, 0.03, 0.04, 0.05, 0.06]:
          epsilon = epsilon_factor * perimeter
          approx = cv2.approxPolyDP(contour, epsilon, closed=True)

          if len(approx) == 4:
              # Reshape to (4, 2)
              quad = approx.reshape(4, 2).astype(np.float64)
              return order_corners_clockwise(quad)

      # Fallback: use 4 corners from convex hull or minAreaRect
      hull = cv2.convexHull(contour)
      if len(hull) == 4:
          return order_corners_clockwise(hull.reshape(4, 2).astype(np.float64))

      # Last resort: use minAreaRect
      rect = cv2.minAreaRect(contour)
      box = cv2.boxPoints(rect)
      return order_corners_clockwise(box.astype(np.float64))


  def order_corners_clockwise(quad):
      """Order vertices clockwise starting from top-left."""
      # Find centroid
      centroid = quad.mean(axis=0)

      # Calculate angles from centroid
      angles = np.arctan2(quad[:, 1] - centroid[1], quad[:, 0] - centroid[0])

      # Sort by angle
      order = np.argsort(angles)
      quad_sorted = quad[order]

      # Rotate to start from top-left (minimum distance from origin)
      distances = np.sum(quad_sorted**2, axis=1)
      top_left_idx = np.argmin(distances)
      quad_ordered = np.roll(quad_sorted, -top_left_idx, axis=0)

      return quad_ordered


  def detect_quads(image_path, model_path, conf_threshold=0.6, imgsz=640):
      """Run YOLO inference and extract quad coordinates."""
      from ultralytics import YOLO

      # Load model
      model = YOLO(model_path)

      # Run inference
      results = model.predict(
          image_path,
          imgsz=imgsz,
          conf=conf_threshold,
          verbose=False
      )

      result = results[0]

      # Check if any detections exist
      if result.masks is None or len(result.masks.data) == 0:
          return [], []

      quads = []
      confidences = []

      # Process each detection
      for i, (mask_xy, conf) in enumerate(zip(result.masks.xy, result.boxes.conf)):
          contour = mask_xy.cpu().numpy()

          # Fit quadrilateral from contour
          quad = fit_quad_from_contour(contour)

          if quad is not None:
              quads.append(quad)
              confidences.append(float(conf.cpu().numpy()))

      return quads, confidences


  def main():
      args = parse_args()

      # Validate inputs
      if not Path(args.image_path).exists():
          print(f"ERROR: Image not found: {args.image_path}", file=sys.stderr)
          sys.exit(1)

      if not Path(args.model_path).exists():
          print(f"ERROR: Model not found: {args.model_path}", file=sys.stderr)
          sys.exit(1)

      try:
          # Run detection
          quads, confidences = detect_quads(
              args.image_path,
              args.model_path,
              args.conf,
              args.imgsz
          )

          # Output format: number of detections followed by quad data
          print(len(quads))

          for quad, conf in zip(quads, confidences):
              # Flatten quad to single line: x1 y1 x2 y2 x3 y3 x4 y4 confidence
              coords = ' '.join([f"{x:.6f} {y:.6f}" for x, y in quad])
              print(f"{coords} {conf:.6f}")

      except Exception as e:
          print(f"ERROR: {str(e)}", file=sys.stderr)
          sys.exit(1)


  if __name__ == "__main__":
      main()
  ```
- [ ] **Rationale:** Dedicated Python script provides clean interface for MATLAB. Outputs to stdout for easy parsing. Matches MATLAB's existing detection logic (quad fitting, corner ordering).
- [ ] **Test:** Run manually from command line:
  ```bash
  python detect_quads.py path/to/image.jpg models/yolo11n_micropad_seg.pt --conf 0.6
  ```
  Verify output format matches specification

---

### 3.2 Update cut_micropads.m to Use Python Inference Helper
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\matlab_scripts\cut_micropads.m`
- [ ] **Task:** Replace direct Python API calls with subprocess call to `detect_quads.py`
- [ ] **Changes:**
  - Lines 1602-1662: Replace `ensurePythonSetup()` function with simplified version:
    ```matlab
    function ensurePythonSetup(pythonPath)
        persistent setupComplete
        if ~isempty(setupComplete) && setupComplete
            return;
        end

        try
            % Check environment variable first
            envPath = getenv('MICROPAD_PYTHON');
            if ~isempty(envPath)
                pythonPath = envPath;
            end

            % Validate Python path is provided
            pythonPath = char(pythonPath);
            if isempty(pythonPath)
                error('cut_micropads:python_not_configured', ...
                    'Python path not configured! Set MICROPAD_PYTHON environment variable or pass pythonPath parameter.');
            end

            if ~isfile(pythonPath)
                error('cut_micropads:python_missing', ...
                    'Python executable not found at: %s', pythonPath);
            end

            fprintf('Python configured: %s\n', pythonPath);
            setupComplete = true;
        catch ME
            setupComplete = [];
            rethrow(ME);
        end
    end
    ```
  - Lines 1672-1685: Delete `loadYOLOModel()` and `getYOLOModel()` functions (no longer needed)
  - Lines 1687-1695: Delete `modelCache()` function (no longer needed)
  - Lines 1767-1830: Replace `detectQuadsYOLO()` function:
    ```matlab
    function [quads, confidences] = detectQuadsYOLO(img, confThreshold, inferenceSize)
        % Run YOLO detection via Python helper script

        % Save image to temporary file
        tmpDir = tempdir;
        tmpImgPath = fullfile(tmpDir, 'micropad_detect_temp.jpg');
        imwrite(img, tmpImgPath, 'JPEG', 'Quality', 95);

        % Get Python path and model path from config
        % (Assuming these are accessible via persistent storage or passed as arguments)
        persistent cachedPythonPath cachedModelPath
        if isempty(cachedPythonPath)
            % First call - get from environment or default
            envPath = getenv('MICROPAD_PYTHON');
            if ~isempty(envPath)
                cachedPythonPath = envPath;
            else
                error('cut_micropads:python_not_configured', 'Python path not configured');
            end
        end
        if isempty(cachedModelPath)
            % Use global config or default model path
            cachedModelPath = 'models/yolo11n_micropad_seg.pt';
        end

        % Build command
        scriptPath = fullfile(fileparts(mfilename('fullpath')), '..', 'python_scripts', 'detect_quads.py');
        cmd = sprintf('"%s" "%s" "%s" "%s" --conf %.2f --imgsz %d', ...
            cachedPythonPath, scriptPath, tmpImgPath, cachedModelPath, ...
            confThreshold, inferenceSize);

        % Run detection
        [status, output] = system(cmd);

        % Clean up temp file
        if isfile(tmpImgPath)
            delete(tmpImgPath);
        end

        if status ~= 0
            error('cut_micropads:detection_failed', 'Python detection failed: %s', output);
        end

        % Parse output
        lines = splitlines(output);
        lines = lines(~cellfun(@isempty, lines));  % Remove empty lines

        if isempty(lines)
            quads = [];
            confidences = [];
            return;
        end

        numDetections = str2double(lines{1});

        if numDetections == 0 || isnan(numDetections)
            quads = [];
            confidences = [];
            return;
        end

        quads = zeros(numDetections, 4, 2);
        confidences = zeros(numDetections, 1);

        for i = 1:numDetections
            if i+1 > length(lines)
                break;
            end

            parts = str2double(split(lines{i+1}));
            if length(parts) < 9
                continue;
            end

            % Parse: x1 y1 x2 y2 x3 y3 x4 y4 confidence
            quad = reshape(parts(1:8), 2, 4)';  % 4x2 matrix
            quads(i, :, :) = quad;
            confidences(i) = parts(9);
        end

        % Filter out empty detections
        validMask = confidences > 0;
        quads = quads(validMask, :, :);
        confidences = confidences(validMask);
    end
    ```
  - Lines 1832-1920: Delete `fitQuadFromContourPoints()` and related helper functions (now handled by Python)
- [ ] **Rationale:** Clean separation - MATLAB calls Python subprocess instead of using Python API directly. Simpler MATLAB code, easier to maintain Python detection logic separately.
- [ ] **Test:** Run `cut_micropads()` with `useAIDetection=true` and verify:
  - Detection works same as before
  - No MATLAB-Python API errors
  - Detected quads have correct format

---

### 3.3 Update cut_micropads.m Configuration for Python Script Path
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\matlab_scripts\cut_micropads.m`
- [ ] **Task:** Store Python script path in configuration for reuse
- [ ] **Changes:**
  - Line 260: Add after model path resolution:
    ```matlab
    % Resolve Python script path
    cfg.pythonScriptPath = fullfile(projectRoot, 'python_scripts', 'detect_quads.py');
    ```
  - Update `detectQuadsYOLO()` function signature to accept cfg parameter:
    ```matlab
    function [quads, confidences] = detectQuadsYOLO(img, cfg)
    ```
  - Update all calls to `detectQuadsYOLO()` to pass `cfg` instead of individual parameters (lines 415, 1068)
- [ ] **Rationale:** Centralized configuration reduces parameter passing and makes script path configurable
- [ ] **Test:** Verify detection still works after configuration refactoring

---

## Phase 4: Update Documentation

### 4.1 Update README.md - Pipeline Architecture Section
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\README.md`
- [ ] **Task:** Update pipeline description to reflect MATLAB-Python separation
- [ ] **Changes:**
  - Lines 261-275: Update "Data Augmentation" section to remove YOLO label export:
    ```markdown
    ### **Why Augmentation is Critical**

    The final Android application will use AI-based **auto-detection** to locate test zones in smartphone photos. Augmentation **multiplies the training dataset** by 5-10x without requiring additional physical experiments, generating diverse viewpoints, lighting conditions, and backgrounds.

    **MATLAB augmentation** creates synthetic image transformations following the standard 1→2→3 pipeline structure. **Python scripts** then prepare YOLO training labels from the MATLAB-generated coordinates.
    ```
  - Lines 383-395: Update "Input/Output Structure" to show label generation in Python:
    ```markdown
    **Inputs:**
    - `1_dataset/{phone}/` - Original smartphone images
    - `2_micropads/{phone}/coordinates.txt` - Polygon vertices (required)
    - `3_elliptical_regions/{phone}/coordinates.txt` - Ellipse parameters (optional)

    **Outputs:**
    - `augmented_1_dataset/{phone}/` - Full synthetic scenes
    - `augmented_2_micropads/{phone}/con_{N}/` - Transformed concentration regions + coordinates.txt
    - `augmented_3_elliptical_regions/{phone}/con_{N}/` - Transformed elliptical patches + coordinates.txt (if input ellipses exist)

    **YOLO Training Labels:** Generated by Python scripts from MATLAB coordinates (see Python pipeline documentation)
    ```
  - Lines 402-408: Update "Integration with ML Pipeline":
    ```markdown
    **For Polygon Detection (YOLO, Faster R-CNN)**
    ```python
    # 1. Generate augmented data in MATLAB
    augment_dataset('numAugmentations', 5)

    # 2. Prepare YOLO dataset with Python (reads MATLAB coordinates, creates labels)
    python python_scripts/prepare_yolo_dataset.py

    # 3. Train model
    python python_scripts/train_yolo.py
    ```
    ```
- [ ] **Rationale:** Users need to understand new workflow where MATLAB generates data, Python prepares training labels
- [ ] **Test:** Read updated sections and verify instructions are clear

---

### 4.2 Update CLAUDE.md - Pipeline Architecture Section
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\CLAUDE.md`
- [ ] **Task:** Document separation of concerns in orchestration guidelines
- [ ] **Changes:**
  - After line 290: Add new section:
    ```markdown
    ### MATLAB-Python Separation of Concerns

    **MATLAB Responsibilities (Data Processing):**
    - Stage pipeline (1→2→3 and augmented variants)
    - Image transformations and geometry
    - Coordinate file generation (coordinates.txt)
    - Feature extraction for ML training

    **Python Responsibilities (AI Model Training/Inference):**
    - YOLO label generation (reads MATLAB coordinates, creates normalized polygon labels)
    - Model training and evaluation
    - Inference helper scripts (called by MATLAB for predictions)

    **Interface:**
    - MATLAB calls Python subprocess (`detect_quads.py`) for AI predictions
    - Python reads MATLAB coordinates (`coordinates.txt`) for label generation
    - No direct MATLAB-Python API coupling (subprocess-based communication)

    **Key Principle:** MATLAB scripts should be agnostic to AI model training details. AI training format changes (e.g., switching from YOLO to Faster R-CNN) should only require Python script modifications, not MATLAB changes.
    ```
- [ ] **Rationale:** Orchestration agent needs to understand architectural boundaries when delegating tasks
- [ ] **Test:** Verify section is readable and principles are clear

---

### 4.3 Update AGENTS.md - Repository Guidelines
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\AGENTS.md`
- [ ] **Task:** Add section on MATLAB-Python separation
- [ ] **Changes:**
  - After line 8: Add new section:
    ```markdown
    ## MATLAB-Python Architecture
    - **MATLAB scripts** (`matlab_scripts/`): Data processing pipeline (stages 1→2→3, augmentation, feature extraction). Should NOT contain AI training logic (no YOLO label export, no model-specific code).
    - **Python scripts** (`python_scripts/`): AI model training/inference (`prepare_yolo_dataset.py`, `train_yolo.py`, `detect_quads.py`). Reads MATLAB coordinates, generates training labels, provides inference helpers.
    - **Communication**: MATLAB calls Python via subprocess (`detect_quads.py` for inference). Python reads MATLAB `coordinates.txt` for label generation. No direct API coupling.
    - **Principle**: Separation of concerns - MATLAB for data, Python for AI. Changes to AI training format should not require MATLAB modifications.
    ```
  - Line 20: Update augmentation example to remove YOLO labels parameter:
    ```markdown
    - Augmentation: `matlab -batch "addpath('matlab_scripts'); augment_dataset('numAugmentations',5);"`
    ```
  - After line 20: Add Python workflow:
    ```markdown
    - Prepare YOLO dataset: `python python_scripts/prepare_yolo_dataset.py`
    - Train YOLO: `python python_scripts/train_yolo.py --stage 1`
    ```
- [ ] **Rationale:** All agents need to understand separation of concerns when making changes
- [ ] **Test:** Verify guidelines are clear and examples are accurate

---

### 4.4 Update .claude/agents/matlab-coder.md
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\.claude\agents\matlab-coder.md`
- [ ] **Task:** Add constraint about AI training concerns
- [ ] **Changes (find CRITICAL IMPLEMENTATION RULES section and add):**
  ```markdown
  - **NEVER add AI training logic to MATLAB scripts** - No YOLO label export, model-specific formats, or training pipeline code. MATLAB focuses on data processing (1→2→3 pipeline, coordinate generation). Python handles all AI training concerns.
  ```
- [ ] **Rationale:** Prevents future violations of separation of concerns
- [ ] **Test:** Read updated rules and verify they're clear

---

### 4.5 Update .claude/agents/python-coder.md
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\.claude\agents\python-coder.md`
- [ ] **Task:** Document Python responsibilities for AI pipeline
- [ ] **Changes (add new section at appropriate location):**
  ```markdown
  ## Python-MATLAB Integration

  **Python handles all AI training and inference:**
  - Reading MATLAB coordinates from `coordinates.txt` files
  - Generating AI training labels (YOLO, Faster R-CNN, etc.)
  - Model training and evaluation
  - Providing inference helper scripts callable from MATLAB

  **Interface with MATLAB:**
  - Subprocess-based: MATLAB calls Python scripts via `system()` command
  - Input: MATLAB coordinates in standardized format (10-column for polygons, 7-column for ellipses)
  - Output: AI-specific label formats (YOLO segmentation, COCO JSON, etc.)

  **Key Principle:** Python owns AI training pipeline. MATLAB should never generate AI training labels directly - Python reads MATLAB coordinates and converts to appropriate AI format.

  **Example workflow:**
  ```
  1. MATLAB: Generate augmented data → augmented_1_dataset/, augmented_2_micropads/coordinates.txt
  2. Python: Read coordinates → Generate YOLO labels → augmented_1_dataset/[phone]/labels/
  3. Python: Train model → models/yolo11n_micropad_seg.pt
  4. MATLAB: Call Python inference helper → detect_quads.py → polygon predictions
  ```
  ```
- [ ] **Rationale:** Python coder needs to understand its role in AI pipeline
- [ ] **Test:** Verify documentation is clear and accurate

---

### 4.6 Update .claude/agents/matlab-code-reviewer.md
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\.claude\agents\matlab-code-reviewer.md`
- [ ] **Task:** Add review criterion for AI training separation
- [ ] **Changes (add to review checklist):**
  ```markdown
  - **Separation of Concerns:** MATLAB scripts should NOT contain AI training logic (no YOLO label export, no model format code). Flag any AI training concerns that should be moved to Python.
  ```
- [ ] **Rationale:** Reviewer should catch violations of architectural principles
- [ ] **Test:** Verify review criterion is actionable

---

### 4.7 Update .claude/agents/python-code-reviewer.md
- [ ] **File:** `C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\.claude\agents\python-code-reviewer.md`
- [ ] **Task:** Add review criterion for MATLAB coordinate reading
- [ ] **Changes (add to review checklist):**
  ```markdown
  - **MATLAB Coordinate Reading:** Python scripts that read MATLAB coordinates should handle standard formats correctly (10-column for polygons, 7-column for ellipses). Verify robust parsing with error handling.
  ```
- [ ] **Rationale:** Python scripts must correctly interface with MATLAB data
- [ ] **Test:** Verify review criterion is clear

---

## Phase 5: Integration Testing and Validation

### 5.1 Test MATLAB Augmentation Pipeline
- [ ] **Task:** Verify augmented pipeline produces expected outputs without YOLO labels
- [ ] **Test Commands:**
  ```matlab
  % In MATLAB
  cd('C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\matlab_scripts')
  augment_dataset('numAugmentations', 1, 'phones', {'iphone_11'})
  ```
- [ ] **Expected Results:**
  - `augmented_1_dataset/iphone_11/` contains synthetic images
  - `augmented_2_micropads/iphone_11/coordinates.txt` exists with 10-column format
  - `augmented_3_elliptical_regions/iphone_11/coordinates.txt` exists (if source ellipses present)
  - NO `labels/` subdirectories created in augmented_1_dataset
- [ ] **Validation:** Check file structure matches expectations, no errors in MATLAB console

---

### 5.2 Test Python Label Generation
- [ ] **Task:** Verify Python script correctly generates YOLO labels from MATLAB coordinates
- [ ] **Test Commands:**
  ```bash
  cd C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis
  python python_scripts\prepare_yolo_dataset.py
  ```
- [ ] **Expected Results:**
  - `augmented_1_dataset/iphone_11/labels/` directory created
  - Label files (`.txt`) match images in `augmented_1_dataset/iphone_11/images/`
  - Label format: `0 x1 y1 x2 y2 x3 y3 x4 y4` (normalized coordinates)
  - Console shows "Generated N label files (M polygons)"
- [ ] **Validation:** Open random label file and verify format, check polygon count matches expectations

---

### 5.3 Test MATLAB-Python Inference Interface
- [ ] **Task:** Verify cut_micropads.m successfully calls detect_quads.py for predictions
- [ ] **Test Commands:**
  ```matlab
  % In MATLAB
  cd('C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\matlab_scripts')
  cut_micropads('numSquares', 7, 'useAIDetection', true, 'phones', {'iphone_11'})
  ```
- [ ] **Expected Results:**
  - Python detection runs without errors
  - Console shows "AI detected N regions (avg confidence: X.XX)"
  - Detected polygons displayed in interactive GUI
  - User can manually adjust and save regions
- [ ] **Validation:** Check detection quality matches previous implementation, no subprocess errors

---

### 5.4 Test End-to-End Workflow
- [ ] **Task:** Run complete workflow from augmentation through training preparation
- [ ] **Test Commands:**
  ```matlab
  % Step 1: Generate augmented data (MATLAB)
  augment_dataset('numAugmentations', 2, 'phones', {'iphone_11'})
  ```
  ```bash
  # Step 2: Prepare YOLO dataset (Python)
  python python_scripts/prepare_yolo_dataset.py

  # Step 3: Verify training can start (don't run full training)
  python python_scripts/train_yolo.py --help
  ```
- [ ] **Expected Results:**
  - All steps complete without errors
  - YOLO dataset ready for training (train.txt, val.txt, labels/)
  - No functionality lost compared to previous workflow
- [ ] **Validation:** Spot-check outputs at each stage, verify data consistency

---

## Progress Tracking

### Overall Status
- [ ] Phase 1: Refactor augment_dataset.m (0/3 tasks)
- [ ] Phase 2: Enhance Python Scripts (0/3 tasks)
- [ ] Phase 3: Create Python Inference Helper (0/3 tasks)
- [ ] Phase 4: Update Documentation (0/7 tasks)
- [ ] Phase 5: Integration Testing (0/4 tasks)

### Key Milestones
- [ ] MATLAB augmentation pipeline cleaned (no AI training code)
- [ ] Python scripts handle all YOLO label generation
- [ ] MATLAB-Python inference interface functional
- [ ] All documentation updated and accurate
- [ ] End-to-end workflow validated

---

## Notes & Decisions

### Design Decisions
- **Why subprocess instead of MATLAB-Python API?** Cleaner separation of concerns, easier to maintain Python code independently, no version compatibility issues between MATLAB and Python environments.
- **Why keep detectQuadsYOLO in cut_micropads.m?** Function name provides continuity, but implementation is now subprocess-based rather than direct API calls.
- **Why generate labels in prepare_yolo_dataset.py?** Centralizes AI training data preparation in Python, ensures labels always synchronized with MATLAB coordinates.

### Known Limitations
- Subprocess communication has slight overhead compared to direct API (negligible for interactive use)
- Python script path must be correctly configured (handled via cfg.pythonScriptPath)
- Temporary image file created for each detection (cleaned up automatically)

### Future Improvements
- [ ] Add JSON output format support for alternative AI frameworks (Faster R-CNN, Mask R-CNN)
- [ ] Implement batch detection mode (process multiple images in single Python call)
- [ ] Add Python unit tests for detect_quads.py
- [ ] Consider migrating cut_micropads.m AI detection to fully external Python script (no embedded logic)

---

## Contact & Support
**Project Lead:** Veysel Y. Yilmaz
**Last Updated:** 2025-11-01
**Version:** 1.0.0
