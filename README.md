# AI Chemistry Video Request Service

A FastAPI backend that turns a learner's chemistry question into a short narrated, captioned educational
video through an asynchronous, resumable job pipeline. Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Full design and decision log: [`docs/superpowers/specs/2026-09-13-chem-video-service-design.md`](docs/superpowers/specs/2026-09-13-chem-video-service-design.md).

Supported queries (one topic pack each, `app/topics/<topic_id>/`):

- How does the pH scale work?
- Why do atoms form covalent bonds?
- What is the difference between ionic and covalent bonding?

## Generated samples

Each folder in [`samples/`](samples/) holds the video, the exact `query.txt`, `storyboard.json`,
`quality_report.json`, `manifest.json`, captions, thumbnail and contact sheet from one real run
(Gemini 2.5 Flash writer, Flash-Lite critic and ASR, Gemini TTS, Manim 1280×720@30), produced once each on
2026-09-15 and committed as generated.

| Sample | Length | Billable cost (ledger) | Writer attempts | Critic (acc/rel/clar/align) | Wall time |
|---|---|---|---|---|---|
| [`ph_scale`](samples/ph_scale/) | 94.2 s | $0.0391 | 3 (G2 rejected length and one claim) | 5/5/5/5 | 580 s |
| [`covalent_bonds`](samples/covalent_bonds/) | 100.9 s | $0.0323 | 1 | 5/5/4/4 | 324 s |
| [`ionic_vs_covalent`](samples/ionic_vs_covalent/) | 103.1 s | $0.0363 | 2 | 5/5/5/5 | 356 s |

Costs come from the `cost_entries` ledger (usage metadata; ASR is estimated). About 57–75 % of each video's
cost is TTS. Rendering is local; its production compute estimate is recorded as non-billable, but CPU time
of the render subprocess is not measurable on Windows, so it shows as $0 in these manifests.

## Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg (with libass) on `PATH`, and the Cairo
libraries Manim needs (`libcairo2-dev pkg-config` on Debian/Ubuntu; bundled wheels on Windows/macOS).

```bash
uv sync --group dev
cp .env.example .env          # set GOOGLE_API_KEY (Gemini); budgets default to $0.20/job, $5 project cap
```

Fonts (Inter, SIL OFL) are bundled in `assets/fonts/`.

## Run

```bash
uv run uvicorn app.main:app          # API on :8000 with an embedded worker thread
uv run python -m app.worker_main     # optional extra worker processes (set RUN_EMBEDDED_WORKER=false on the API)
uv run python -m app.cost.report     # project spend from the cost ledger
```

State lives in `data/` (SQLite database and ledger, work directories, artifacts, TTS cache).

## API (`/v1`)

| Method & path | Purpose |
|---|---|
| `POST /v1/videos` `{"query": "..."}` (+ optional `Idempotency-Key`) | request a video → `202` job with `Location` |
| `GET /v1/videos?status=&topic_id=&limit=&cursor=` | list jobs, newest first |
| `GET /v1/videos/{id}` | status, stage, progress, events, quality summary, cost, artifacts, stage durations |
| `GET /v1/videos/{id}/content` | the MP4 (`409 VIDEO_NOT_READY` until it succeeds) |
| `GET /v1/videos/{id}/artifacts[/{name}]` | published files: video, thumbnail, captions, storyboard, quality report, contact sheet, manifest |
| `POST /v1/videos/{id}/retry` | re-queue a failed, retryable job |
| `GET /v1/topics` · `GET /healthz` | supported topics · DB, ffmpeg, worker heartbeat, project spend |

Errors always use `{"error": {"code", "message", "retryable", "details"}}`. Interactive docs: `/docs`.

```bash
curl -s -X POST localhost:8000/v1/videos -H 'Content-Type: application/json' \
  -d '{"query": "How does the pH scale work?"}'
curl -s localhost:8000/v1/videos/<id>            # poll until "status": "succeeded"
curl -s -o video.mp4 localhost:8000/v1/videos/<id>/content
```

## Tests

```bash
uv run pytest -q -m "not integration and not render and not live"   # fast suite (~40 s)
uv run pytest -q -m integration        # ffmpeg: assembly, media QA, orchestrator over fakes (~2 min)
uv run pytest -q -m render             # real Manim templates and renderer (~2 min)
RUN_LIVE=1 uv run pytest -q -m live    # PAID Gemini calls; check the spend report first
uv run ruff check . && uv run ruff format --check . && uv run mypy app
```

The orchestrator tests run the whole pipeline through `build_container(fake_providers=True)` (curated
storyboard LLM, tone TTS, colour-card renderer), including resume from checkpoints, stale leases, render
and QA failures, and publish recovery.
