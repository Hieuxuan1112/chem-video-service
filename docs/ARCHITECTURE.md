# Architecture

The authoritative design is `docs/superpowers/specs/2026-09-13-chem-video-service-design.md`. This note
summarises the three boundaries the challenge asks about.

```
client ──HTTP──▶ FastAPI (/v1)  ──▶ JobRepository (SQLite)  ◀── Worker (embedded thread or process)
                                                                     │ lease + heartbeat + fencing
                                                                     ▼
                                                               VideoPipeline
      writer/critic ─▶ voicing ─▶ render plan ─▶ renderer ─▶ assembler ─▶ media QA ─▶ ArtifactStore
      (LLMProvider)    (TTS/ASR)   (pure code)    (Manim)     (ffmpeg)     (ffmpeg)    (local files)
```

## 1. Job lifecycle

- `POST /v1/videos` resolves the query to a topic pack (keyword rules; unsupported queries get
  `422 UNSUPPORTED_TOPIC`) and inserts a `queued` job. An optional `Idempotency-Key` replays the same job.
- A worker claims the oldest runnable job atomically and holds a **lease** (60 s) renewed by a heartbeat
  thread. Every write the pipeline makes is **fenced** by `(job_id, worker_id, attempt)`; a worker that
  lost its lease cannot write state, checkpoints or artifacts.
- `running` jobs move through stages `scripting → reviewing → voicing → rendering → quality_check →
  publishing`. Each transition is an event (`GET /v1/videos/{id}` returns them with stage-based progress).
- Outcomes: `succeeded` after publication; on an error the job gets a classified `error_code`/`cause`.
  Retryable causes (provider outage, render crash/timeout, transient ffmpeg, I/O) are rescheduled with
  backoff (60 s, 300 s) up to 3 claims; deterministic causes (template/layout bug, QA rule violation,
  invalid curated fallback) fail immediately. `POST /v1/videos/{id}/retry` re-queues a failed retryable job.
- A crashed worker's job is reclaimed after lease expiry and **resumes** from its checkpoints.

## 2. Persistence and artifact boundary

- **SQLite** (WAL) holds `jobs`, `job_events`, `stage_checkpoints`, `worker_heartbeats`, and the cost
  ledger (`cost_entries`, `budget_state`). It is the single source of truth for job state; any number of
  worker processes can share it. In production this maps to Postgres or Cloud SQL without changing the
  repository interface.
- **Checkpoints** (`app/pipeline/checkpoints.py`): each stage writes to `work/<job>/attempt-<n>/<stage>/`
  and promotes a DB row in a fenced transaction. A checkpoint is reused only if the row exists, the
  `stage.json` hash matches the row, every file hash matches, the input hash matches, and all versions
  (pipeline, schema, prompts, topic pack, renderer, voice) match. A retry therefore never pays for the
  same storyboard or narration twice and never uses a stale or partial output.
- **ArtifactStore** (`app/artifacts/store.py`, protocol + `LocalArtifactStore`): files are copied to a
  temporary name and `os.replace`d; `manifest.json` (sha256, bytes, versions, models, voice, cost,
  checkpoint attempts, stage durations) is written **last**, so a valid manifest means publication is
  complete. Published files: `video.mp4`, `thumbnail.jpg`, `captions.vtt`, `storyboard.json`,
  `quality_report.json`, `contact_sheet.jpg`, `manifest.json`. The API serves only this allow-list. A GCS
  bucket with generation preconditions and signed URLs is the production mapping.

## 3. AI and video-generation boundary

The pipeline depends on small protocols, never on SDKs:

| Protocol | Implementations | Notes |
|---|---|---|
| `LLMProvider` | `GeminiLLMProvider`, `FakeLLM`, `CuratedFakeLLM` | JSON-schema output, writer and critic |
| `TTSProvider` | `GeminiTTSProvider` (primary), `EdgeTTSProvider` (free fallback), `FakeTTS` | |
| `ASRProvider` | `GeminiASRProvider`, `FakeASR` | report-only word error rate |
| `Renderer` | `ManimRenderer` (isolated subprocess), `FakeRenderer` | consumes a renderer-agnostic `RenderPlan` |
| `ArtifactStore` | `LocalArtifactStore` | |

Every paid call goes through `SpendGuard`: reserve the projected cost against the per-job ($0.20) and
project ($5 hard cap, $3 warning) budgets **before** calling, then settle the actual cost from usage
metadata. `build_container(fake_providers=True)` wires the fakes for tests and demos without a key.

**Why the LLM never draws numbers.** The LLM writes a closed storyboard (narration, on-screen titles and
visual *choices* from enums). Every chemistry value on screen (pH of a substance, electron shells,
shared pairs, transferred electrons, log ratios) is resolved by code from `app/topics/chemistry_data.py`,
and scenes are drawn by deterministic Manim templates. Generative video models were rejected on cost and
repeatability.

## 4. Reliability under non-determinism

| Layer | Gate | On failure |
|---|---|---|
| Storyboard | G1 schema/bounds, G2 topic rules (required facts, misconceptions, allowed visuals, text fits its layout slot), G3 critic scores | located feedback to the writer, bounded retries, then the topic's curated storyboard (`generation_source=curated_fallback`) |
| Narration | A1–A4 per clip (decodes, duration vs words, loudness, internal silence), A5 ASR word error rate | re-synthesise the clip, then switch the whole video to the fallback voice |
| Render | per-scene result files, exact frame counts, safe-area and text-fit assertions | crash/timeout re-rendered once; template/layout errors are non-retryable bugs |
| Assembly | sample-accurate narration placement, ±1 frame repair | transient ffmpeg failures retried once |
| Media QA | Q1 container, Q2 durations, Q3 loudness/silence, Q4 black/freeze, Q5 blank or duplicated scenes | deterministic rule failure fails the job with the full report |

After voicing everything is deterministic, so identical inputs produce the same video; retries and resumes
converge to one consistent artifact.
