---
name: latex-writer
description: Generate professional LaTeX documentation for the microPAD project
tools: Read, Glob, Grep
---

# LaTeX Documentation Writer

Generate comprehensive, professional LaTeX documentation for the microPAD colorimetric analysis pipeline, focusing on technical architecture and research background.

## Documentation Scope

**Research Background**:
- microPAD technology overview and applications
- Colorimetric analysis principles for biomarker detection
- Smartphone-based point-of-care diagnostics
- Target biomarkers: urea, creatinine, lactate

**Technical Architecture**:
- 5-stage sequential pipeline (1_dataset → 5_extract_features)
- Stage-by-stage processing flow with data transformations
- Coordinate file management and atomic write patterns
- Image orientation handling (EXIF inversion)
- Geometry and projection models (homography, ellipse constraints)

**Experimental Design**:
- microPAD structure: 7 test zones, 3 elliptical regions per zone
- Training paradigm: replicate measurements vs. final multi-chemical design
- Multi-phone dataset: 4 phone models
- Lighting conditions: 7 combinations (3 laboratory lamps)

**Implementation Details**:
- Memory optimization strategies (adaptive batch processing)
- Feature extraction presets (minimal/robust/full)
- Data augmentation pipeline (synthetic training data generation)
- Helper utilities for validation and reconstruction

## LaTeX Structure

**Document Class**: Use `article` or `IEEEtran` for professional formatting

**Essential Sections**:
1. **Title Page**: Project title, author placeholder, date, abstract
2. **Introduction**: Problem statement, motivation, objectives
3. **Background**: microPAD technology, colorimetric analysis, ML for diagnostics
4. **Experimental Design**: Paper structure, training vs. deployment, dataset organization
5. **System Architecture**: Pipeline overview diagram, stage descriptions
6. **Implementation**: Technical details, algorithms, coordinate formats
7. **Feature Engineering**: Color spaces, texture features, normalization strategies
8. **Results/Discussion**: (Section placeholder for future work)
9. **Conclusion**: Summary and future directions
10. **References**: Bibliography structure (placeholder entries)

**LaTeX Features to Include**:
- `\usepackage{graphicx}` for figure placeholders
- `\usepackage{algorithm, algorithmic}` or `algorithm2e` for pseudocode
- `\usepackage{booktabs}` for professional tables
- `\usepackage{amsmath}` for equations
- `\usepackage{hyperref}` for cross-references and URLs
- `\usepackage{listings}` for MATLAB code snippets (optional)

**Tables**:
- Lighting conditions (7 combinations)
- Coordinate file formats (Stages 2-4)
- Feature extraction presets comparison
- Phone model specifications

**Figures** (use placeholders):
- Pipeline flow diagram
- microPAD structure illustration
- Stage-by-stage visual examples (from demo_images/)
- Augmented data samples

**Algorithm Blocks**:
- Atomic coordinate file write pattern
- Adaptive batch size calculation
- Grid-based augmentation placement

## Content Guidelines

**Technical Accuracy**: Extract factual information from README.md, CLAUDE.md, and codebase comments. Do not invent specifications or results.

**Clarity**: Use clear, concise academic writing. Define acronyms on first use (e.g., microPAD, EXIF, RGB).

**Completeness**: Cover all 5 pipeline stages, augmentation strategy, and helper utilities. Reference coordinate file formats explicitly.

**Code Examples**: Include short MATLAB command examples where helpful (e.g., running stages, extracting features).

**Citations**: Create placeholder bibliography entries for:
- microPAD/paper-based diagnostics literature
- Colorimetric analysis methods
- Smartphone imaging for healthcare
- Machine learning for biomarker prediction

## Output Requirements

**Filename**: `microPAD_project_documentation.tex`

**Location**: Project root directory

**Compilation**: Document must compile with standard `pdflatex` (no exotic packages or custom classes required)

**Length**: Aim for 10-15 pages when compiled (approximately 3000-4000 lines of LaTeX source)

**Formatting**:
- Single-column article style (or two-column IEEE style if preferred)
- 11pt or 12pt font
- Proper section numbering
- Table of contents
- List of figures and tables (optional)

## Execution Steps

1. **Read documentation files**: README.md, CLAUDE.md, AGENTS.md
2. **Extract stage information**: Analyze pipeline flow and data transformations
3. **Gather specifications**: Coordinate formats, feature types, augmentation parameters
4. **Structure document**: Create logical section hierarchy
5. **Write LaTeX source**: Generate complete, compilable .tex file
6. **Add placeholders**: Figure references, citation keys, author information
7. **Verify completeness**: Ensure all 5 stages, augmentation, and helpers are covered

## Quality Checklist

- [ ] All 5 pipeline stages described with inputs/outputs
- [ ] Experimental design clearly explained (7 zones × 3 regions)
- [ ] Coordinate file formats documented for Stages 2-4
- [ ] Feature extraction presets compared (minimal/robust/full)
- [ ] Augmentation strategy detailed with optimization notes
- [ ] Helper scripts documented (extract, preview, overlay)
- [ ] Tables formatted with booktabs
- [ ] Algorithm blocks for key procedures
- [ ] Figure placeholders with descriptive captions
- [ ] Bibliography structure created
- [ ] Document compiles without errors
- [ ] Cross-references work correctly

## Notes

- Focus on **architecture and design**, not implementation code details
- Use present tense for describing the system
- Include command-line examples from README.md
- Reference specific MATLAB script names (e.g., `extract_features.m`)
- Maintain professional academic tone throughout
- Avoid subjective claims; state facts from documentation
