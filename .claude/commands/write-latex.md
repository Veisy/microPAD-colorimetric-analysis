# write-latex

Generate professional LaTeX documentation for the microPAD colorimetric analysis project

## Usage

```
/write-latex [style]
```

## Description

Launches the LaTeX Documentation Writer agent to create comprehensive technical documentation covering:
- Research background (microPAD technology, colorimetric analysis, biomarker detection)
- Technical architecture (5-stage pipeline, data flow, coordinate management)
- Experimental design (7 test zones, 3 regions, multi-phone dataset, lighting conditions)
- Implementation details (memory optimization, feature extraction, augmentation)

The agent generates a complete, compilable LaTeX document ready for conversion to PDF.

## Parameters

**style** (optional): Document formatting style
- `article` - Standard academic article format (default)
- `ieee` - IEEE conference/transaction format (two-column)
- `report` - Extended report format with chapters

## Examples

```bash
# Generate standard article-style documentation
/write-latex

# Generate IEEE-style documentation
/write-latex ieee

# Generate extended report format
/write-latex report
```

## Output

**File**: `microPAD_project_documentation.tex` (project root)

**Content includes**:
1. Title page and abstract
2. Introduction and problem statement
3. Background on microPAD technology and colorimetric analysis
4. Experimental design (paper structure, training paradigm, dataset)
5. System architecture (5 pipeline stages with flow diagrams)
6. Implementation details (algorithms, coordinate formats, optimizations)
7. Feature engineering (color spaces, texture, normalization)
8. Conclusion and future work
9. Bibliography structure (placeholder entries)

**Features**:
- Professional formatting with proper sections and numbering
- Tables for lighting conditions, coordinate formats, feature presets
- Algorithm blocks for key procedures (atomic writes, batch sizing, augmentation)
- Figure placeholders for pipeline diagrams and visual examples
- Cross-references and hyperlinks
- MATLAB command examples from README.md

**Length**: 10-15 pages when compiled (3000-4000 lines of LaTeX)

## Compilation

Compile the generated LaTeX file to PDF:

```bash
# Standard compilation
pdflatex microPAD_project_documentation.tex

# With bibliography (if using BibTeX)
pdflatex microPAD_project_documentation.tex
bibtex microPAD_project_documentation
pdflatex microPAD_project_documentation.tex
pdflatex microPAD_project_documentation.tex
```

**Requirements**: Standard LaTeX distribution (TeX Live, MiKTeX, MacTeX)

**Packages used**: graphicx, algorithm, algorithmic, booktabs, amsmath, hyperref, listings

## Use Cases

- **Academic documentation**: Technical reports, project documentation, thesis chapters
- **Conference submissions**: Adapt generated content for paper submissions
- **Project presentations**: Extract sections for slides and presentations
- **Team onboarding**: Comprehensive reference for new team members
- **Grant proposals**: Technical background and methodology sections

## Customization

After generation, you can customize:
- Author information and affiliations (line ~10-15)
- Figure paths to point to actual images in `demo_images/`
- Bibliography entries with real citations
- Abstract and conclusion to reflect specific results
- Section emphasis based on audience (more background vs. more technical detail)

## Tips

**Before running**:
- Ensure all documentation files (README.md, CLAUDE.md) are up to date
- Review `demo_images/` folder for available figures
- Decide on target audience (technical vs. general)

**After generation**:
- Replace `\includegraphics` placeholders with actual image paths
- Add real citation entries if referencing literature
- Adjust section emphasis based on document purpose
- Proofread technical details and specifications

**Converting to PDF**:
- Use online converters: Overleaf, Papeeria (upload .tex file)
- Local compilation: Install TeX Live or MiKTeX
- VS Code: Install LaTeX Workshop extension
- Command line: `pdflatex microPAD_project_documentation.tex`

## Related Commands

- `/review-matlab` - Review MATLAB code quality before documenting
- `/commit-changes` - Commit documentation to version control
