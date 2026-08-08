# Agent Constraints and Guidelines

This document outlines the mandatory constraints and guidelines that any AI agent must follow when working on this repository. These constraints are derived from the developer's explicit requirements during the development process.

## Git Configuration Requirements

### Author Identity
All commits MUST be authored solely by the developer. Never use generic profiles like "Developer" or any other identity.

**Required Git Configuration:**
```bash
git config --global user.name "mahajialirezaei"
git config --global user.email "m.a.hajialirezaei05@gmail.com"
```

**Commit Authorship Rules:**
- Every commit must show only `mahajialirezaei` as both author and committer
- Never allow co-authorship or attribution to generic profiles (e.g., "and Developer")
- If a commit shows multiple authors, it must be amended using:
  ```bash
  git commit --amend --reset-author
  git push --force
  ```

## Security Constraints

### Token Management
- **NEVER** share GitHub tokens, API keys, or credentials in any documentation, commit messages, or code comments
- Tokens provided for authentication purposes must be used only for the immediate task and never stored in version control
- Do not log or echo tokens in command outputs
- If a token is accidentally exposed, it must be rotated immediately

### File Restrictions
- Never commit sensitive files containing credentials
- Always check `.gitignore` before adding new configuration files

## Branch Management

### Branch Naming Convention
- Feature branches: `feature/<description>`
- Bugfix branches: `bugfix/<description>`
- Refactor branches: `refactor/<description>`
- All feature/bugfix/refactor branches must be created from `develop`
- All merges must go through pull requests to `develop` first
- Only stable, tested code should be merged to `main`

### Commit Message Standards
- Use conventional commit format: `<type>: <description>`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Messages should be clear, concise, and in imperative mood
- Reference issue numbers or phases when applicable

## Documentation Requirements

### Required Documentation Files
- `BRANCH.md`: Must always list all active and completed branches with descriptions
- `DATA_STRUCTURE.md`: Must be consulted before using images or backgrounds
- `FIXING-REQ.md`: Contains tasks that must be implemented
- `README.md`: Project overview and usage instructions
- `TODO.md`: Pending tasks and future work

### Updating BRANCH.md
When creating new branches or completing features:
1. Add the branch to BRANCH.md with a clear description
2. Include location of key files
3. List main contents/deliverables
4. Mark status as completed when merged

## Data Usage Guidelines

### Image and Background Assets
- Always consult `DATA_STRUCTURE.md` before using images or backgrounds
- Use only approved directories:
  - Clean scans: `data/clean_scans/`
  - Backgrounds: `data/random_backgrounds/`
  - Raw data: `data/raw/`
- Respect the naming conventions and organization defined in DATA_STRUCTURE.md

## Implementation Priorities

### Task Execution Order
1. Read and understand `FIXING-REQ.md` completely before implementation
2. Consult relevant documentation (PDF docs, DATA_STRUCTURE.md)
3. Implement tasks in the order specified
4. Test each component before proceeding
5. Update documentation after implementation

### Quality Standards
- Code must be production-ready before merging
- All inference scripts must be executable by TAs without modification
- Evaluation metrics must match specifications exactly
- Follow existing code style and patterns

## Communication Protocol

### Progress Reporting
- Report completion status clearly
- List specific files modified or created
- Confirm branch names and merge status
- Verify author information is correct before pushing

### Error Handling
- If a constraint cannot be met, report immediately
- Do not proceed with operations that violate security constraints
- Request clarification when requirements are ambiguous

---

## Summary of Critical Constraints

| Constraint | Priority | Enforcement |
|------------|----------|-------------|
| Single author identity (mahajialirezaei) | CRITICAL | Every commit |
| Never share tokens | CRITICAL | Always |
| Use develop as base branch | HIGH | Branch creation |
| Update BRANCH.md | HIGH | After branch creation |
| Consult DATA_STRUCTURE.md | MEDIUM | Before data usage |
| Conventional commits | MEDIUM | Every commit |

**Note:** Violation of these constraints, especially security and authorship rules, requires immediate correction via commit amendment or credential rotation.
