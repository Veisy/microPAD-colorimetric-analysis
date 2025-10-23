# microPAD colorimetric analysis

A MATLAB-based colorimetric analysis pipeline for microfluidic paper-based analytical devices (microPADs). Processes smartphone images of test strips to extract color features and predict biomarker concentrations (urea, creatinine, lactate) using machine learning.

![Pipeline Overview](demo_images/stage1_original_image.jpeg)

## Table of Contents
- [Requirements](#requirements)
- [Dataset Structure](#dataset-structure)
- [Pipeline Stages](#pipeline-stages)
- [Quick Start](#quick-start)
- [Data Augmentation](#data-augmentation)
- [Directory Layout](#directory-layout)
- [Tips and Troubleshooting](#tips-and-troubleshooting)

---

## Requirements

- **MATLAB R2020a or later** with Image Processing Toolbox
- Run scripts from the project folder or `matlab_scripts/` folder

---

## Dataset Structure

### Phone Models & Lighting Conditions

The dataset is organized by **phone model** (e.g., `iphone_11/`, `samsung_a75/`, `realme_c55/`). Each phone captures **7 images per microPAD**, representing different lighting combinations using 3 laboratory lamps:

| Lighting ID | Lamp 1 | Lamp 2 | Lamp 3 | Description |
|-------------|--------|--------|--------|-------------|
| **light_1** | ✓ | ✗ | ✗ | Only lamp 1 |
| **light_2** | ✗ | ✓ | ✗ | Only lamp 2 |
| **light_3** | ✗ | ✗ | ✓ | Only lamp 3 |
| **light_4** | ✓ | ✓ | ✗ | Lamps 1 + 2 |
| **light_5** | ✓ | ✗ | ✓ | Lamps 1 + 3 |
| **light_6** | ✗ | ✓ | ✓ | Lamps 2 + 3 |
| **light_7** | ✓ | ✓ | ✓ | All lamps |

This lighting variation helps the model work well under different real-world lighting conditions.

### microPAD Structure

Each microPAD paper strip contains **7 test zones** (concentrations), and each test zone has **3 elliptical measurement regions**:

- **Final design:** The 3 regions will contain different chemicals (urea, creatinine, lactate)
- **Training phase:** All 3 regions contain the same chemical at the same concentration, providing 3 replicate measurements per concentration level
- This allows separate datasets to be collected for training machine learning models for each chemical

---

## Pipeline Stages

The pipeline processes images through **5 sequential stages**. Each stage reads from folder `N_*` and writes to `(N+1)_*`:

### **Stage 1 → 2: Extract microPAD Paper**

**Script:** `crop_micropad_papers.m`

**Input:** Raw smartphone photos in `1_dataset/{phone_model}/`
**Output:** Cropped paper strips in `2_micropad_papers/{phone_model}/`

![Stage 1 to 2](demo_images/stage2_micropad_paper.jpeg)

Isolates the microPAD paper from the background using interactive rotation and cropping.

---

### **Stage 2 → 3: Cut Concentration Regions**

**Script:** `cut_concentration_rectangles.m`

**Input:** Cropped papers from `2_micropad_papers/`
**Output:** Individual concentration rectangles in `3_concentration_rectangles/{phone_model}/con_{N}/`

![Stage 2 to 3](demo_images/stage3_concentration_rectangle.jpeg)

Cuts out individual test zones from each paper strip (default: 7 zones per strip).

---

### **Stage 3 → 4: Extract Elliptical Patches**

**Script:** `cut_elliptical_regions.m`

**Input:** Concentration rectangles from `3_concentration_rectangles/`
**Output:** Elliptical patches in `4_elliptical_regions/{phone_model}/con_{N}/`

![Stage 3 to 4](demo_images/stage4_elliptical_region_1.jpeg)

Extracts three elliptical measurement regions from each test zone. In the final microPAD design, these three regions will contain different chemicals (urea, creatinine, lactate). For training purposes, each experiment fills all three regions with the same concentration of a single chemical, providing three replicate measurements per concentration level.

---

### **Stage 4 → 5: Feature Extraction**

**Script:** `extract_features.m`

**Input:**
- Concentration rectangles from `3_concentration_rectangles/` (used as paper/white reference)
- Elliptical patches from `4_elliptical_regions/`

**Output:** Excel feature tables in `5_extract_features/`

Measures color values from each test zone and adjusts them based on the white paper background. Saves results to Excel files for machine learning.

**Example usage:**
```matlab
extract_features('preset','robust','chemical','lactate')
```

**Available presets:**
- `minimal`: Essential color ratios and basic statistics
- `robust`: Comprehensive feature set (recommended)
- `full`: All available features
- `custom`: User-defined feature groups

**What gets measured** (robust preset extracts 80+ features):

- **Color relative to paper**: How much the test zone color differs from the white paper
- **Color ratios**: Red/Green, Red/Blue, Green/Blue ratios that work under different lighting
- **Basic color statistics**: Average, median, and variation of RGB, HSV, and Lab color values
- **Texture patterns**: Stripe patterns, uniformity, roughness
- **Color gradients**: How color changes across the test zone
- **Other measurements**: Edge sharpness, color intensity ranges

---

## Quick Start

Run these commands **in order** from the project folder:

### Full Pipeline (from raw images)

```bash
# Stage 1: Crop microPAD papers
matlab -batch "addpath('matlab_scripts'); crop_micropad_papers;"

# Stage 2: Cut concentration rectangles (7 regions per strip)
matlab -batch "addpath('matlab_scripts'); cut_concentration_rectangles;"

# Stage 3: Extract elliptical regions
matlab -batch "addpath('matlab_scripts'); cut_elliptical_regions;"

# Stage 4: Extract features (reads both 3_ and 4_; change 'lactate' to your chemical name)
matlab -batch "addpath('matlab_scripts'); extract_features('preset','robust','chemical','lactate');"
```

### Typical Workflow (starting from Stage 2)

If you already have cropped papers in `2_micropad_papers/`:

```bash
matlab -batch "addpath('matlab_scripts'); cut_concentration_rectangles;"
matlab -batch "addpath('matlab_scripts'); cut_elliptical_regions;"
matlab -batch "addpath('matlab_scripts'); extract_features('preset','robust','chemical','lactate');"
```

---

## Data Augmentation

**Script:** `augment_dataset.m` (optional)

Generates synthetic training data by transforming Stage 2 outputs (papers and polygons) and composing them onto procedural backgrounds. If Stage 4 ellipse coordinates exist, augmented ellipse patches are also produced.

![Augmented Dataset](demo_images/augmented_dataset_1.jpg)
*Synthetic scene with transformed microPAD*

![Augmented Concentration](demo_images/augmented_concentration_rectangle_1.jpeg)
*Augmented concentration region*

![Augmented Ellipse](demo_images/augmented_elliptical_region1.jpeg)
*Augmented elliptical patch*

**What it does:**
- Shared perspective + rotation per paper
- Random non-overlapping placement using spatial grid acceleration
- Procedural backgrounds: uniform, speckled, laminate (white/black), skin
- Moderate-density distractor artifacts (1-20 per image)
- Color-safe photometric augmentation (brightness/contrast, white balance, saturation/gamma)
- Optional motion blur or Gaussian blur
- Outputs 1 original (aug_000) + N augmented versions per paper

**Usage:**
```matlab
% Generate 5 augmented versions per original image (optimized for speed)
augment_dataset('numAugmentations', 5, 'rngSeed', 42)

% Optional parameters (defaults shown)
augment_dataset('numAugmentations', 5, ...
                'photometricAugmentation', true, ...
                'blurProbability', 0.25, ...
                'motionBlurProbability', 0.15, ...
                'occlusionProbability', 0.0, ...
                'independentRotation', false)
```

**Performance:**
- ~3x faster than previous version (1.0s vs 3.0s per augmented image)
- Grid-based placement with automatic background expansion
- Simplified texture generation and polygon warping

**Inputs and outputs:**
- Reads from `2_micropad_papers/` and `3_concentration_rectangles/` (coordinates required)
- If `4_elliptical_regions/coordinates.txt` is present, also writes ellipse patches

**Output folders:**
- `augmented_1_dataset/` (synthetic scenes)
- `augmented_2_concentration_rectangles/` (transformed concentration regions)
- `augmented_3_elliptical_regions/` (transformed elliptical patches)

If ellipse coordinates are unavailable, run Stage 3 on `augmented_2_concentration_rectangles/` to generate augmented ellipse patches.

---

## Helper Scripts

Three utility scripts in `matlab_scripts/helper_scripts/` for recreating images and checking quality:

### **extract_images_from_coordinates.m**

Recreates all processed images from `coordinates.txt` files. Instead of saving thousands of image files, you only need to keep small text files with coordinates.

**Why this matters:**
- You don't need to save processed images (stages 2-4) - just keep the `coordinates.txt` files
- Anyone can recreate the exact same images from the coordinates
- Saves storage space (gigabytes → kilobytes)

**Usage:**
```matlab
addpath('matlab_scripts/helper_scripts');
extract_images_from_coordinates();
```

Run this script to recreate any missing processed images from your saved coordinates.

---

### **preview_overlays.m**

Opens a viewer that shows how well your coordinates match the actual images. Displays rectangles, polygons, and ellipses overlaid on the original photos.

**Use this to:**
- Check if your annotations are accurate
- Find mistakes before extracting features
- Verify your work after editing `coordinates.txt` files

**Usage:**
```matlab
addpath('matlab_scripts/helper_scripts');
preview_overlays();  % Press 'n' to navigate, 'q' to quit
```

---

### **preview_augmented_overlays.m**

Same as `preview_overlays.m`, but for checking the augmented (synthetic) dataset quality.

**Usage:**
```matlab
addpath('matlab_scripts/helper_scripts');
preview_augmented_overlays();
```

---

## Directory Layout

```
microPAD-colorimetric-analysis/
├── matlab_scripts/          # Main processing scripts
│   ├── crop_micropad_papers.m
│   ├── cut_concentration_rectangles.m
│   ├── cut_elliptical_regions.m
│   ├── extract_features.m
│   ├── augment_dataset.m
│   └── helper_scripts/      # Utility functions
├── 1_dataset/               # Raw smartphone photos
│   ├── iphone_11/
│   ├── iphone_15/
│   ├── realme_c55/
│   └── samsung_a75/
├── 2_micropad_papers/       # Cropped paper strips
├── 3_concentration_rectangles/  # Concentration regions + coordinates.txt
├── 4_elliptical_regions/    # Elliptical patches + coordinates.txt
├── 5_extract_features/      # Feature tables (.xlsx)
├── augmented_1_dataset/     # (Optional) Synthetic scenes
├── augmented_2_concentration_rectangles/
├── augmented_3_elliptical_regions/
└── demo_images/             # Visual examples for documentation
```

**Note:** The processed images are not saved to version control (only the `coordinates.txt` files are saved).

---

## Tips and Troubleshooting

### Coordinate Files

Each stage saves a `coordinates.txt` file with position and shape information:
- **Stage 2:** Paper location and rotation
- **Stage 3:** Corner points of each test zone
- **Stage 4:** Ellipse center, size, and rotation

**If corrupted:** Delete the file and re-run the stage to create a new one.

### Memory Issues

If MATLAB runs out of memory, process fewer images at once:

```matlab
extract_features('preset', 'robust', 'batchSize', 10)
```

### Running from the Wrong Folder

If you see warnings about missing folders, make sure you're running scripts from:
- The main project folder, OR
- The `matlab_scripts/` folder

### Testing Before Full Processing

Before processing all images, test with 2-3 sample images to make sure everything works correctly.

---

## Output Example

Sample rows from `5_extract_features/robust_lactate_t0_features.xlsx` (showing subset of 80+ feature columns):

| PhoneType  | ImageName                   | RG_ratio | RB_ratio | GB_ratio | L       | a       | b      | delta_E_from_paper | R_paper_ratio | Label |
|------------|-----------------------------|----------|----------|----------|---------|---------|--------|--------------------|---------------|-------|
| iphone_11  | IMG_0957_aug_000_con_0.jpeg | 0.461449 | 0.360812 | 0.781910 | 53.9623 | 10.3421 | 18.952 | 32.0361            | 0.7234        | 0     |
| iphone_11  | IMG_0957_aug_000_con_0.jpeg | 0.460265 | 0.365116 | 0.793272 | 54.0647 | 10.1204 | 19.124 | 31.5070            | 0.7189        | 0     |
| iphone_15  | IMG_1234_aug_000_con_3.jpeg | 0.483148 | 0.388192 | 0.803463 | 55.1563 | 11.8934 | 20.456 | 30.2742            | 0.7512        | 3     |
| realme_c55 | IMG_5678_aug_000_con_5.jpeg | 0.492301 | 0.395421 | 0.810234 | 56.2341 | 12.4567 | 21.345 | 29.1234            | 0.7634        | 5     |
| samsung_a75| IMG_9012_aug_000_con_6.jpeg | 0.501234 | 0.402345 | 0.821345 | 57.3456 | 13.2345 | 22.456 | 28.3456            | 0.7789        | 6     |

**Notes:**
- Each row represents one elliptical region measurement (replicate)
- Multiple rows with the same concentration indicate replicate measurements from the 3 elliptical regions within that test zone
- `Label` column shows the known concentration (0-6) for training the model
- During training, all 3 replicates per test zone contain the same chemical at the same concentration
- Full table has 80+ columns (only some shown above)
- Can split data into separate training and testing files
