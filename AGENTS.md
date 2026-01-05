# Repository Guidelines

## Project Structure & Module Organization
The source lives under `src/` with feature modules split by area:
`src/config/` (settings), `src/auth/`, `src/health/`, `src/mcp/`,
`src/metrics/`, `src/schemas/`, and `src/libs/` (shared utilities).
Tests are expected under `tests/` (currently empty). The project is
configured with `pyproject.toml`, and dependencies are locked in `uv.lock`.

## Build, Test, and Development Commands
This repo is set up for the `uv` workflow (see `Dockerfile`).

- `uv sync --dev` - create a local virtual environment with dev tools.
- `uv run pytest` - run the test suite (looks under `tests/`).
- `uv run ruff check src tests` - lint Python sources.
- `uv run black .` - format code to the configured style.
- `uv run pyright` - run static type checks.

Docker builds use the included `Dockerfile`. It expects a root-level
`run_server.py` entrypoint (not currently present).

## Coding Style & Naming Conventions
Python 3.11+ is required. Formatting follows Black with a line length of 88.
Linting is done with Ruff. Type checks use Pyright. Use `snake_case` for
functions and variables, `PascalCase` for classes, and keep module names
short and descriptive (e.g., `vector_database.py`).

## Testing Guidelines
Pytest is configured with `test_*.py` or `*_test.py` file naming and
`Test*` classes. Async tests use `pytest-asyncio` with `asyncio_mode=auto`.
If you add tests, place them under `tests/` and prefer fast unit tests.
Markers are available: `unit`, `integration`, `slow`.

## Commit & Pull Request Guidelines
Git history is minimal (only an initial commit), so no convention is
established. Use clear, imperative commit messages (or Conventional
Commits if you prefer). For PRs, include:
- a short summary of changes,
- test results or rationale if tests are not added,
- configuration changes and any required environment variables.

## Security & Configuration Tips
Configuration is expected via environment variables (`python-dotenv` is
included). Keep secrets out of git, and document required variables in
the PR description when adding new configuration.
