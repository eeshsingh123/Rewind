# Rules

## Design

- Open/closed first. A new chat channel, segmenter backend, language model, storage, rule predicate or sampling strategy is a new module that implements an existing port or registry entry. It must not require edits to the core.
- Dependencies point inward: `app -> adapters -> core -> contracts`. The core depends only on `contracts` and the ports it defines. The core never imports an SDK, `httpx`, `modal`, `boto3`, `cv2` or `fastapi`. `import-linter` enforces this in `poe check`.
- Ports are `typing.Protocol` classes owned by the core. Adapters implement them. Only the composition root (`app/`) builds adapters and passes them in. No DI framework, no global singletons, no service locators.
- One reason to change per module. If a change request touches more than two modules, revisit the boundary before coding.
- Abstract only at real seams: an external boundary (a model API, storage, a chat channel, ffmpeg), or a second concrete implementation. No speculative interfaces, options or config.
- Extension points are registries keyed by a discriminator (rule predicates, sampling strategies, channels). Adding an entry is additive; the code that dispatches through the registry never changes.
- Reuse before writing. Delete before adding.

## Code

- Python 3.12, uv, ruff (lint + format), pyright (strict for `contracts` and `core`), pytest, import-linter. One command runs them all: `uv run poe check`.
- No module docstrings and no file header comments. Docstrings only on a port method whose contract the signature can't say, and then one line.
- Comments explain a non-obvious why in one line. Never restate the code, never narrate history, no TODOs without an owner and a track.
- Plain, conventional names. No clever, abbreviated or joke names. No `Manager`/`Helper`/`Utils`/`common` dumping grounds.
- Small functions, early returns, no dead code, no commented-out code.
- Types on every public boundary; no `Any` at a boundary. Pydantic validates at the edges: webhooks, HTTP bodies, model API responses, language model JSON, files read from disk. Inside the core, trust the data. Contracts are frozen models.
- Money is `Decimal`. Time is seconds as `float` from the start of the footage. Wall-clock time is a timezone-aware UTC `datetime`.
- Async at I/O edges (`httpx.AsyncClient`, FastAPI). CPU work (OpenCV, ffmpeg) runs in a worker or `asyncio.to_thread`, never on the event loop.
- Errors are handled where something can be done about them; otherwise they propagate. Each port has one error type that adapters raise. No silent catches; no bare `except`.
- Every paid call goes through the cost ledger and the budget guard. No adapter calls a paid API any other way.
- Secrets come only from settings (the environment, or `.env` locally). They are never logged, never put in URLs, never committed.
- Tests are few and focused: core behaviour at module boundaries, and flows that cross layers. One test per rule, not per example. Adapters are tested against recorded real responses in `tests/recordings/`, never against live APIs in CI. No tests for what pyright or Pydantic already proves, and none added only for coverage.

## Git

- `main` is protected. Work happens on `feat/`, `fix/`, `chore/`, `docs/` branches.
- Every branch is cut from an up-to-date `main` and every PR targets `main`. No stacked PRs.
- Hackathon exception (D-11): tracks run in parallel, one branch and one PR per track, as long as they touch different packages. Contracts change only in their own PR, merged before any track that depends on them.
- No force-push and no rebase on a pushed branch. To update a branch, merge `main` into it.
- Each sub-module is its own series of commits, each one logical change, so the PR can be reviewed commit by commit.
- The owner reviews and merges every PR. The agent never merges.
- Commit messages: imperative subject under 72 characters, with a body only when the why is not obvious.

## Definition of done

1. The track works end to end and was actually run.
2. `uv run poe check` passes.
3. The diff was re-read against this file.
4. DECISIONS.md and MEMORY.md reflect any new decision or state.
