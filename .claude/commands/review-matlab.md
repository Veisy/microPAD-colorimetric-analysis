# review-matlab

Review MATLAB code for quality, style, and best practices

## Usage

```
/review-matlab [file_pattern]
```

## Description

Launches the MATLAB Code Reviewer agent to analyze code quality, maintainability, and adherence to MATLAB and project-specific best practices.

The agent performs comprehensive reviews covering:
- Code structure and readability
- Error handling and input validation
- Performance issues (array growth, pre-allocation)
- Documentation quality and accuracy
- MATLAB idioms and built-in function usage
- Project-specific conventions (naming, pipeline integration)
- Mask-aware feature computation patterns

## Examples

```bash
# Review a specific script
/review-matlab extract_features.m

# Review all helper scripts
/review-matlab helper_scripts/*.m

# Review entire matlab_scripts directory
/review-matlab matlab_scripts/
```

## Output

The agent provides a structured review with:
1. Overall assessment
2. Critical issues affecting functionality
3. Documentation violations with exact quotes
4. Code quality concerns
5. MATLAB best practice recommendations
6. Prioritized suggestions (High/Medium/Low)

## Related Commands

- `/optimize-matlab` - Apply performance optimizations
- `/analyze-performance` - Deep performance analysis