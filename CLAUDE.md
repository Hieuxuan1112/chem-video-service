# CLAUDE.md — AI Chemistry Video Request Service

Growtrics/TaloTrace backend coding challenge. **Read `HANDOFF.md` first**, then the spec and the plan.

- Spec (authoritative design): `docs/superpowers/specs/2026-09-13-chem-video-service-design.md` (revision 2)
- Plan (P0–P12, 64 tasks): `docs/superpowers/plans/2026-09-13-chem-video-service.md`

## Non-negotiable rules

1. **The spec is the source of truth.** Do not change an approved decision silently. If implementation reveals a
   problem with the spec, stop, explain it, propose the smallest change, and wait for approval. Then update the spec
   and add an entry to its §15 change log.
2. **No git writes by the agent.** Never run `git add`, `git commit`, `git push`, create branches, merge or rebase.
   The user does all git and GitHub work. Read-only git (`status`, `diff`, `log`) is fine. At each commit
   checkpoint, report the changed files, test output and the plan's suggested commit message.
3. **TDD per the plan's standard task loop.** Tests first, see them fail for the right reason, implement, run the
   task tests plus the fast suite, ruff and mypy.
4. **Never claim something works without running it.** Paste real command output. No fabricated numbers,
   measurements or test results. Unverified external values stay marked as unverified.
5. **Budget.** Every paid Gemini call goes through `SpendGuard` against the project ledger (`data/app.sqlite`):
   - warning at $3.00 (reported once);
   - hard cap at $5.00 (refuse before spending);
   - $0.20 per job.

   Before any paid task, run `uv run python -m app.cost.report`. Never delete `data/app.sqlite` or any
   ledger/state file.
6. **Ask the user first** before downloading files (e.g. the Inter font zip), pulling Docker images, or any
   outward-facing action.
7. **LLM output is never the authority for chemistry facts.** Controlled values come from
   `app/topics/chemistry_data.py`. LLM and user strings are rendered with Manim `Text`, never `MarkupText`.
8. **Language.**
   - Code, comments, commit messages and repository docs are in English.
   - Talk to the user in Vietnamese.
   - Interview-prep docs are Vietnamese and live **outside** the repo in `D:\growtrics-interview-prep\`.
9. Keep changes minimal and on-plan. No extra infrastructure (no Redis/Celery/Postgres/Kafka/k8s).

## Commands

```bash
uv sync --group dev
uv run pytest -q -m "not integration and not render and not live"   # fast suite
uv run pytest -q -m integration        # needs ffmpeg
uv run pytest -q -m render             # needs Manim
RUN_LIVE=1 uv run pytest -q -m live    # PAID — check spend first
uv run ruff check . && uv run ruff format --check . && uv run mypy app
uv run uvicorn app.main:app --reload
uv run python -m app.cost.report
```

## Environment

Windows 11 + Git Bash. Python 3.12.6. uv. ffmpeg 8.1 (libass) on PATH. Docker Desktop.
The Gemini key lives in `D:\langgraph-agent-lab\.env` (`GOOGLE_API_KEY`). Copy it into this repo's `.env` without
printing it, and only after `.gitignore` excludes `.env`.
