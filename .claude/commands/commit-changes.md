# commit-changes

Manage git commits with organized, clear commit messages

## Usage

```
/commit-changes [file_pattern]
```

## Description

Launches the Git Commit Manager agent to analyze changes, organize commits logically, and create clear, professional commit messages that follow project standards.

The agent handles:
- Change analysis via git status/diff
- Logical grouping by change type (bug fixes, features, config, docs)
- File selection and staging
- Commit message creation (imperative mood, specific descriptions)
- Pre-commit validation

## Examples

```bash
# Commit all changes with organized messages
/commit-changes

# Commit specific files
/commit-changes matlab_scripts/extract_features.m

# Commit by pattern
/commit-changes *.md

# Commit agent configurations
/commit-changes .claude/
```

## Commit Categories

The agent groups changes into:
- **Bug fixes**: Corrections to existing functionality
- **Features**: New capabilities or pipeline stages
- **Configuration**: Parameter or constant changes
- **Documentation**: README, CLAUDE.md, comments
- **Performance**: Optimizations without functional changes
- **Cleanup**: Style fixes, formatting, dead code removal

## Commit Message Standards

- Imperative mood ("Fix bug" not "Fixed bug")
- Specific descriptions ("Add polygon cropping validation" not "Update function")
- First line under 50 characters when possible
- **NEVER include any mention of AI, Claude, or automated tools in commit messages**
- **NEVER add "Generated with Claude Code" or similar attribution**
- **NEVER add "Co-Authored-By: Claude" or any AI co-author tags**
- Focus on functional impact, not implementation
- Commit messages should appear as if written by a human developer

## Output

The agent shows proposed commits before execution:
```
Proposed commit 1: "Fix circular mask boundary handling"
Files: extract_features.m, createCircularPatchMask.m

Proposed commit 2: "Update documentation with mask handling details"
Files: CLAUDE.md, README.md
```

## Safety Features

- Validates no unintended files are staged
- Checks for secrets or large files
- Ensures atomic, complete commits
- Prevents broken intermediate states

## Related Commands

- `/review-matlab` - Pre-commit code review
- Standard git commands via Bash tool