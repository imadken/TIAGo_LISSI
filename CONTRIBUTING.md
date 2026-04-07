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
   - Pin and document dependency changes in `requirements.txt`.
   - Explain why a new dependency is required in your PR.

4. **Keep changes focused and minimal**
   - Submit small, reviewable PRs.
   - Avoid unrelated refactors in the same PR.

5. **Document behavior changes**
   - Update docs in `docs/` and/or `README.md` when behavior, setup, or usage changes.

## Pull request process

- Use the PR template in `.github/pull_request_template.md`.
- Provide a clear summary and detailed bullet points of what changed.
- Describe validation performed (tests, manual checks, or rationale when tests are not runnable).
