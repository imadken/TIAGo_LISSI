# Contributing to TIAGo_LISSI

Thank you for contributing.

## Scope and priorities

Most expected contributions are **new skills for the agent** (especially for interns building on this repository).

Other contributions are also welcome, including:
- bug fixes
- generalization/improvements of existing modules
- compatibility with additional LLM providers
- integration with other agentic libraries

## Contribution guidelines

1. **Respect the project structure**
   - Keep Python code under `tiago_lissi/` and place code in the right domain package (`agent/`, `skills/`, `perception/`, `navigation/`, etc.).
   - Keep tests/diagnostic scripts in `tests/`.
   - Keep setup and operational documentation in `docs/`.

2. **Prefer extending existing patterns**
   - For new capabilities, prefer implementing a new skill under `tiago_lissi/skills/`.
   - Reuse existing abstractions instead of duplicating logic.

3. **Manage dependencies carefully**
   - Add new dependencies only when necessary.
   - This project uses **uv** for dependency management. New or changed dependencies must be reflected in `pyproject.toml` (the canonical dependency file). `requirements.txt` is kept only as a first-time bootstrap reference and must not be treated as the primary source of truth after the project is initialized.
   - Explain why a new dependency is required in your PR.

4. **Test your changes**
   - All changes must be validated with tests.
   - Run existing tests in `tests/` to confirm nothing is broken.
   - If your change introduces new behavior that is not covered by the current test suite, add new tests in `tests/` following the patterns already in place.

5. **Respect the uv workflow**
   - This project uses [uv](https://github.com/astral-sh/uv) as the Python package manager.
   - Always use `uv` commands for environment and dependency management (e.g. `uv venv`, `uv pip sync`, `uv add`).
   - Do **not** bypass `uv` by invoking plain `pip install` or manually editing lock files.
   - The `pyproject.toml` file is the authoritative source for project metadata and dependencies; update it (not `requirements.txt`) when adding or removing packages after the project is initialized.

6. **Keep changes focused and minimal**
   - Submit small, reviewable PRs.
   - Avoid unrelated refactors in the same PR.

7. **Document behavior changes**
   - Update docs in `docs/` and/or `README.md` when behavior, setup, or usage changes.

## Pull request process

- Use the PR template in `.github/pull_request_template.md`.
- Provide a clear summary and detailed bullet points of what changed.
- Describe validation performed (tests, manual checks, or rationale when tests are not runnable).
