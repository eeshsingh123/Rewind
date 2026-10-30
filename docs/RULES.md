# Rules

How code is written here. Read before writing code.

## Layout

```
src/rewind/
  contracts/   frozen Pydantic models shared by everything; depends on nothing but pydantic
  core/        domain logic and ports (Protocols); depends only on contracts
    perception/   ingest, idle skip, sampling planner, SAM runner, cache
    rules/        rule language, predicates, engine, zones, compiler
    analysis/     a check end to end: tracks, parts, proof clips, replies
    ask/          a photo or video session: Muse's first look, the agent, its tools, areas, checked claims
    conversation/ the old chat flow with lines and rules (replaced by ask/ in phase 3)
    budget/       ledger and budget guard
  adapters/    one module per external system, each implementing a core port
    segmenters/ models/ storage/ video/ jobs/ channels/
  app/         composition root: settings, wiring, FastAPI routes
sim/           Blender scenario renderer (frozen); depends on contracts only
eval/          scoring (empty until the accuracy table is built)
scripts/       live smoke runs and the tunnel launcher; never run in CI
tests/         focused tests, and recordings/ of real API responses
```

## Design

- Open/closed first. A new chat channel, segmenter, language model, storage, rule predicate or sampling strategy is a new module that implements an existing port or registry entry. It must not require edits to the core.
- Dependencies point inward: `app -> adapters -> core -> contracts`. The core never imports an SDK, `httpx`, `cv2` or `fastapi`. `import-linter` enforces this in `poe check`.
- Ports are `typing.Protocol` classes owned by the core. Only `app/` builds adapters and passes them in. No DI framework, no global singletons.
- One reason to change per module. A change that touches more than two modules: rethink the boundary first.
- Abstract only at real seams (a model API, storage, a chat channel, ffmpeg) or for a second real implementation. No speculative options or config.
- Reuse before writing. Delete before adding.

## Code

- Python 3.12, uv, ruff, pyright (strict for `contracts` and `core`), pytest, import-linter. One command runs them all: `uv run poe check`.
- No module docstrings and no file header comments. Docstrings only on a port method whose contract the signature can't say, in one line.
- Comments explain a non-obvious why in one line. Never restate the code, never narrate history, never point at plan or ticket codes.
- Plain, conventional names. No `Manager`/`Helper`/`Utils`/`common` dumping grounds.
- Small functions, early returns, no dead or commented-out code.
- Types on every public boundary; no `Any` at a boundary. Pydantic validates at the edges (webhooks, HTTP bodies, model responses, files read from disk). Contracts are frozen and forbid unknown fields.
- Money is `Decimal`. Footage time is seconds as `float` from the start. Wall-clock time is timezone-aware UTC.
- Async at I/O edges. CPU work (OpenCV, ffmpeg) runs in a worker or `asyncio.to_thread`, never on the event loop.
- Errors are handled where something can be done about them; otherwise they propagate. No silent catches, no bare `except`.
- Every paid call goes through the ledger and the budget guard.
- Secrets come only from settings (the environment, or `.env`). Never logged, never in URLs, never committed.
- Tests are few and focused: core behaviour and flows that cross layers. Adapters are tested against recordings in `tests/recordings/`, never live APIs in CI. Re-record `tests/recordings/muse/compile_examples.json` (`scripts/smoke_rules.py --record`) whenever the compile prompt or rule schema changes. No tests only for coverage.

## Git

- `main` is protected. Work on `feat/`, `fix/`, `chore/` or `docs/` branches cut from an up-to-date `main`. One PR per piece of work, targeting `main`; no stacked PRs.
- Two sessions may work in parallel only when they touch different packages. A contract change goes in its own PR first.
- No force-push and no rebase on a pushed branch. To update a branch, merge `main` into it.
- Commits are logical steps; imperative subject under 72 characters, a body only when the why is not obvious.
- The owner reviews and merges every PR. The agent never merges.

## Definition of done

1. It works end to end and was actually run.
2. `uv run poe check` passes.
3. The diff was re-read against this file.
4. docs/PLAN.md (and DECISIONS.md, if a decision changed) are updated.
