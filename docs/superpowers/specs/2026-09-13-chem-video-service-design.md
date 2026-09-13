# AI Chemistry Video Request Service — Design Spec

- **Date:** 2026-09-13
- **Status:** Approved in review (Parts 1–4), pending final read-through of this document
- **Scope source:** `Agentic_Backend_Challenge_AI_Chemistry_Video_Request_Service.pdf` (Growtrics coding challenge)
- **Implementation state at time of writing:** nothing implemented. Only throw-away feasibility spikes were run (see §2).

---

## 1. Goal and success criteria

A learner submits a chemistry question; the backend turns it into a short (≈45–100 s) narrated,
animated explanation video as an asynchronous job; the client lists jobs, polls status, and opens the
finished video.

Required queries, supported end-to-end:

| Topic id | Canonical learner query |
|---|---|
| `ph_scale` | How does the pH scale work? |
| `covalent_bonds` | Why do atoms form covalent bonds? |
| `ionic_vs_covalent` | What is the difference between ionic and covalent bonding? |

Priorities, in order: correctness → reliability → production-quality architecture → engineering
practice → extensibility → testing → cost awareness → maintainability. No infrastructure is added
without a strong architectural reason.

The project is done when:

1. Unit, integration and render test suites pass.
2. The chaos evaluation (§10.3) holds all five invariants.
3. The live reliability evaluation (3 topics × 5 runs) has run and its report is committed with the real numbers, whatever they are.
4. Three sample videos (one per topic) with their input query, storyboard, manifest and quality report are committed.
5. README commands have been executed from a clean clone.
6. `docker build` succeeds and the demo runs inside the container.

Development spend on paid APIs is capped at **USD 3.00 in total** (dev + calibration + live eval + samples).

## 2. Evidence gathered before design (spikes, not project code)

| Spike | Result | Design consequence |
|---|---|---|
| `pip install manim` on Windows, Python 3.12 | Installed in ~48 s (manim 0.21.0) | Manim is viable on the dev machine |
| Manim 720p30, 10 s scene with `MarkupText` subscripts | 14 s wall time, ~8 s of it process start-up; H₂O subscript renders without LaTeX | Render all scenes in one subprocess; no LaTeX dependency |
| Frame inspection of that render | Manim fell back to a Windows serif font; RGB interpolation RED→PURPLE produced a muddy gradient; content clustered in the upper half | Bundle a font; designed discrete palette; explicit layout grid |
| `gemini-2.5-flash-preview-tts`, one sentence | 5.53 s audio, ~138 audio tokens → **≈25 audio tokens/s** | TTS cost model |
| `edge-tts` 7.2.8 | Works, MP3 output | Usable as independent fallback voice provider |
| ffmpeg 8.1 (gyan full build) filters | `ass`, `subtitles` (libass), `loudnorm`, `ebur128`, `silencedetect`, `volumedetect`, `blackdetect`, `freezedetect` present | Deterministic media QA with ffmpeg only |
| Starlette 0.50 `FileResponse` | Supports HTTP `Range` | Seekable video playback without custom code |
| Python 3.12 bundled SQLite | 3.45.3 | `UPDATE … RETURNING` available (≥ 3.35) |
| Gemini docs | Audio input = 32 tokens/s | ASR cost model |

## 3. Architecture overview

```
Client (curl / scripts/demo.py / Swagger UI)
  │  POST /v1/videos   GET /v1/videos[/{id}]   GET /v1/videos/{id}/content
  ▼
FastAPI (app/api) ── resolve topic (deterministic) ──► JobRepository (SQLite, WAL)
                                                            ▲
Worker (app/jobs/worker.py) ── claim with lease ────────────┘
  │  heartbeat thread · fencing token (job id + worker id + attempt)
  ▼
Pipeline orchestrator (app/pipeline)
  scripting → reviewing → voicing → rendering → quality_check → publishing
  │            │             │           │
  │  providers/llm       providers/tts  render/ (Renderer protocol → ManimRenderer subprocess,
  │  providers/asr                       MediaAssembler = ffmpeg, MediaQA = ffprobe/ffmpeg)
  ▼
Checkpoints (work/<job_id>/…, DB pointer)  →  ArtifactStore (data/artifacts/<job_id>/)
Cost ledger + event timeline (SQLite) · JSON logs
```

Rejected alternatives (recorded in `docs/DECISIONS.md`):

- **LLM writes Manim code.** Compiles-but-looks-wrong output, sandboxing required, expensive retry loop, unstable across runs.
- **Text-to-video (Veo 3.1) or image-per-slide.** USD 3.75–30 per 75 s video; unverifiable chemistry and garbled labels.
- **Celery/Redis/Kafka.** The SQLite queue gives durability and atomic claiming with zero added infrastructure, and maps directly to Cloud Tasks / Pub/Sub later.
- **FastAPI `BackgroundTasks`.** Loses jobs on restart and cannot bound concurrency.
- **LangGraph for the pipeline.** A fixed stage sequence with one bounded loop gains nothing from a graph runtime, and its checkpointer would duplicate the job store.
- **moviepy.** Python frame loops are slow and opaque; the ffmpeg CLI is deterministic and standard.
- **Prometheus `/metrics`.** Out of scope; the per-job event timeline, JSON logs and the eval report cover observability for this service.

### 3.1 Repository layout

```
chem-video-service/
  app/
    main.py                 app factory, dependency wiring (constructor injection, no registry)
    config.py               pydantic-settings; every threshold in this spec is a named setting
    api/                    routes.py, schemas.py (DTOs), errors.py (error envelope)
    domain/                 job.py (JobStatus, Stage, transition table), errors.py (classification table)
    jobs/                   repository.py (SQLite), worker.py, clock.py (injectable Clock)
    pipeline/               orchestrator.py, checkpoints.py, storyboard.py, writer.py, critic.py,
                            gates/ (g1_schema.py, g2_rules.py, g2_claims.py, g2_layout.py),
                            voicing.py, audio_checks.py, prompts/ (writer_v1.md, rewrite_v1.md, critic_v1.md)
    providers/              retry.py, llm/ (protocol, gemini, fake), tts/ (protocol, gemini, edge, fake),
                            asr/ (protocol, gemini, fake)
    render/                 protocol.py, plan.py (RenderPlan, timeline, beats), layout.py (slots, theme),
                            captions.py, assembler.py, media_qa.py, ffmpeg.py, fake.py,
                            manim/ (runner.py, templates/*.py, theme.py)
    topics/                 registry.py, resolver.py, chemistry_data.py, claims.py,
                            ph_scale/ covalent_bonds/ ionic_vs_covalent/ (pack.yaml, curated_storyboard.json)
    artifacts/              store.py (ArtifactStore protocol, LocalArtifactStore)
    cost/                   pricing.py, ledger.py, budget.py
    observability/         logging.py (JSON), events.py
  assets/fonts/             Inter (OFL) + LICENSE
  tests/                    unit/ integration/ render/ live/ fixtures/ golden/
  evals/                    reliability_eval.py, calibrate_gates.py, results/
  samples/<topic_id>/       video.mp4, query.txt, storyboard.json, manifest.json, quality_report.json, thumbnail.jpg
  scripts/                  demo.py, export_samples.py, make_zip.sh
  docs/                     ARCHITECTURE.md, RELIABILITY.md, COST.md, DECISIONS.md, AI_WORKFLOW.md,
                            API_WALKTHROUGH.md, superpowers/specs/, superpowers/plans/
  Dockerfile  pyproject.toml  uv.lock  .github/workflows/ci.yml  README.md  .env.example
```

Runtime directories `data/` (SQLite DB, artifacts, work, TTS cache) are git-ignored.

## 4. Job lifecycle and persistence (Part 1)

### 4.1 States and stages

```
queued ──claim──► running ──► succeeded
  ▲                 │ │
  │                 │ └──► failed ──POST /retry (retryable only)──► queued
  └─scheduled retry─┘
```

- `running` carries a `stage`: `scripting → reviewing → voicing → rendering → quality_check → publishing`.
  The writer/critic loop alternates `scripting`/`reviewing` per attempt.
- The allowed transitions are exactly the arrows above. They live in one table in `domain/job.py`, and every other pair raises `IllegalTransition`.
- `running → queued` (**scheduled retry**) is used when a stage fails with a retryable classification and the job has scheduled retries left (§4.6).
- Lease reclaim keeps the job in `running` and moves ownership to the new worker with `attempt + 1`.

### 4.2 Schema (SQLite)

```sql
CREATE TABLE jobs (
  id TEXT PRIMARY KEY,                 -- "vid_" + uuid4 hex
  query TEXT NOT NULL,
  topic_id TEXT NOT NULL,
  status TEXT NOT NULL,                -- queued | running | succeeded | failed
  stage TEXT,
  attempt INTEGER NOT NULL DEFAULT 0,  -- monotonic fencing token, never reset
  lease_claims INTEGER NOT NULL DEFAULT 0,  -- claims since last enqueue; reset on retry
  scheduled_retries INTEGER NOT NULL DEFAULT 0,
  worker_id TEXT,
  lease_until REAL,                    -- unix epoch seconds (UTC)
  not_before REAL,                     -- earliest claim time for scheduled retries
  generation_source TEXT,              -- llm | curated_fallback
  fallback_reason TEXT,
  error_code TEXT, error_cause TEXT, error_message TEXT, retryable INTEGER,
  idempotency_key TEXT UNIQUE,
  request_fingerprint TEXT,
  pipeline_version TEXT NOT NULL,
  created_at REAL NOT NULL, updated_at REAL NOT NULL, started_at REAL, finished_at REAL
);
CREATE INDEX jobs_claim ON jobs(status, not_before, created_at);

CREATE TABLE job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, attempt INTEGER NOT NULL,
  ts REAL NOT NULL, type TEXT NOT NULL, stage TEXT, data_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX job_events_job ON job_events(job_id, id);

CREATE TABLE stage_checkpoints (
  job_id TEXT NOT NULL, stage_key TEXT NOT NULL,   -- e.g. "reviewing", "voicing", "rendering/scene-03"
  attempt INTEGER NOT NULL, input_hash TEXT NOT NULL, manifest_sha256 TEXT NOT NULL,
  versions_json TEXT NOT NULL, created_at REAL NOT NULL,
  PRIMARY KEY (job_id, stage_key)
);

CREATE TABLE cost_entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, attempt INTEGER, ts REAL NOT NULL,
  stage TEXT, operation TEXT NOT NULL,             -- writer | critic | tts | asr | render_compute
  provider TEXT NOT NULL, model TEXT, status TEXT NOT NULL,   -- ok | error
  units_json TEXT NOT NULL,                        -- input_tokens, output_tokens, audio_seconds, characters, cpu_seconds
  usd REAL NOT NULL, cost_source TEXT NOT NULL,    -- actual | estimated | free
  pricing_version TEXT NOT NULL
);
CREATE INDEX cost_entries_ts ON cost_entries(ts);
```

Connection settings: `journal_mode=WAL`, `busy_timeout=5000`, `synchronous=NORMAL`,
`foreign_keys=ON`. Each thread uses its own connection.

### 4.3 Atomic claim

The worker runs the sweep, then the claim, once per poll (default poll interval 1 s).

```sql
-- sweep: expired leases that exhausted automatic reclaims
UPDATE jobs SET status='failed', error_code='WORKER_LOST', error_cause='lease_expired_max_claims',
       retryable=1, worker_id=NULL, lease_until=NULL, finished_at=:now, updated_at=:now
WHERE status='running' AND lease_until < :now AND lease_claims >= :max_claims;

-- claim: one statement, atomic under SQLite's single-writer lock
UPDATE jobs
SET status='running', worker_id=:worker, lease_until=:now + :lease_s,
    attempt=attempt+1, lease_claims=lease_claims+1,
    started_at=COALESCE(started_at, :now), updated_at=:now
WHERE id = (
    SELECT id FROM jobs
    WHERE (status='queued' AND (not_before IS NULL OR not_before <= :now))
       OR (status='running' AND lease_until < :now AND lease_claims < :max_claims)
    ORDER BY created_at LIMIT 1)
  AND ((status='queued' AND (not_before IS NULL OR not_before <= :now))
    OR (status='running' AND lease_until < :now AND lease_claims < :max_claims))
RETURNING *;
```

Why this is safe: SQLite serialises writers. When two workers race, the second statement
re-evaluates the outer `WHERE` against the row already updated by the first, matches nothing, and
returns no row. There is no read-then-write gap to exploit.

A reclaimed job emits a `lease_reclaimed` event with the previous `worker_id` and `attempt`.

Defaults: `lease_s=60`, `heartbeat_interval_s=15`, `max_claims=3`.

### 4.4 Heartbeat and fencing

- While a job runs, a heartbeat thread executes
  `UPDATE jobs SET lease_until=:now+:lease_s, updated_at=:now WHERE id=:id AND worker_id=:w AND attempt=:a AND status='running'`.
- Every worker-side write (stage changes, events, checkpoint pointers, terminal state) uses the same fencing predicate. A `rowcount` of 0 raises `StaleLeaseError`.
- On heartbeat failure or `StaleLeaseError`, the worker sets an in-process `lease_lost` flag and logs `lease_lost`. It cannot write a DB event, because its fence is no longer valid. The orchestrator checks the flag between steps; the render/ffmpeg subprocess watcher kills the process tree (psutil) and abandons the job **without writing anything else**.
- Before publishing, the worker renews the lease so publication (sub-second) happens well inside a fresh lease. Residual risk (lease lost mid-publish) is documented: publication is idempotent and derived only from DB-authoritative checkpoints (§6), so a concurrent publisher writes identical bytes.

### 4.5 Idempotency

- `POST /v1/videos` accepts an optional `Idempotency-Key` header (1–128 visible ASCII characters).
- `request_fingerprint = sha256(normalized JSON body)`.
- Insert with the key. On `UNIQUE` conflict, load the existing row:
  - Same fingerprint → `202` with the original job body and header `Idempotent-Replayed: true`.
  - Different fingerprint → `409 IDEMPOTENCY_KEY_REUSED`.
- Keys never expire in this prototype (documented).
- No deduplication by query content: two learners asking the same question are two valid requests, and repeated-run evaluation depends on it.

### 4.6 Failure classification and retries

Every failure is an `AppError(code, cause, retryable, message, details)`. A single table in
`domain/errors.py` maps `(code, cause)` to `retryable`, and a table-driven test covers every entry.
Unknown exceptions map to `INTERNAL_ERROR / unclassified / retryable=false`.

| Code | Cause | Retryable |
|---|---|---|
| `WORKER_LOST` | `lease_expired_max_claims` | true |
| `CURATED_FALLBACK_INVALID` | `deterministic_bug` | false |
| `TTS_PROVIDER_UNAVAILABLE` | `provider_outage` | true |
| `TTS_VALIDATION_FAILED` | `output_invalid_after_retries` | true |
| `RENDER_FAILED` | `subprocess_crash` | true |
| `RENDER_TIMEOUT` | `timeout` | true |
| `RENDER_TEMPLATE_ERROR` | `deterministic_bug` | false |
| `RENDER_LAYOUT_ERROR` | `deterministic_bug` | false |
| `ASSEMBLY_FAILED` | `ffmpeg_transient` | true |
| `ASSEMBLY_FAILED` | `invalid_input` | false |
| `QUALITY_CHECK_FAILED` | `io_error` | true |
| `QUALITY_CHECK_FAILED` | `deterministic_output_bug` | false |
| `ARTIFACT_STORE_ERROR` | `io_error` | true |
| `INTERNAL_ERROR` | `unclassified` | false |

Retry layers, innermost first:

1. **Provider call:** 429 / 5xx / timeout → exponential backoff with full jitter (base 1 s, factor 2, cap 20 s, max 3 attempts). 401 / 403 / 400 → no retry.
2. **Stage-local:** TTS clip re-synthesis (§7.3), one scene re-render (§8.2), one assembly re-run (§8.4).
3. **Scheduled job retry:** a retryable stage failure moves the job `running → queued` with `not_before = now + [60 s, 300 s][scheduled_retries]` while `scheduled_retries < 2`; it resumes from checkpoints (§6). When exhausted, the job moves to `failed`.
4. **Manual retry:** `POST /v1/videos/{id}/retry` on `failed` + `retryable=true` executes `UPDATE … SET status='queued', lease_claims=0, scheduled_retries=0, error_*=NULL WHERE id=:id AND status='failed' AND retryable=1`. If two retries race, exactly one succeeds; the other gets `409 NOT_RETRYABLE`.

LLM content failures never fail a job; they route to the curated fallback (§5.6).

## 5. Content generation (Part 2)

Principle: **the LLM may express facts, but it is never the authority for facts.** Topic packs
and `chemistry_data.py` are authoritative. The LLM decides how to explain; code verifies what was
said and supplies every controlled numeric value used in visuals.

### 5.1 Topic packs

Each `app/topics/<topic_id>/pack.yaml` is validated by a Pydantic model at start-up:

- `id`, `subject`, `title`, `version`, `canonical_query`.
- `match`: `all_of` keyword groups, `any_of`, `none_of`.
- `required_facts[]`: `id`, `statement`, `detect` (keyword groups over the normalized full narration).
- `misconceptions[]`: `id`, `pattern` (regex), `explanation`.
- `claim_checks[]`: names of checkers in `topics/claims.py`.
- `allowed_visuals[]`, `required_visuals[]`, `allowed_elements[]`, `allowed_molecules[]`, `allowed_compounds[]`.
- `pronunciations`: display form → spoken form, applied to TTS input only.
- `curated_storyboard.json`.

Adding a STEM topic means adding a pack. A new template or claim checker is needed only if the topic needs a new visual or claim type.

Required facts and constraints for the three packs:

| Topic | Required facts | Required visuals | Misconceptions (examples) |
|---|---|---|---|
| `ph_scale` | scale runs 0–14; below 7 acidic; 7 neutral; above 7 basic/alkaline; logarithmic, one unit = 10×; measures hydrogen-ion (H⁺) concentration | `ph_scale` | "pH 7 is acidic", "higher pH means more acidic", "linear scale", "each unit is twice" |
| `covalent_bonds` | atoms share electron pairs; to reach a full/stable outer shell (octet, duet for H); typically between non-metals; shared pair attracted by both nuclei holds atoms together; a named example | `electron_sharing` | "covalent bonds transfer electrons", "atoms lose electrons to form covalent bonds" |
| `ionic_vs_covalent` | ionic: electrons transferred metal → non-metal; oppositely charged ions attract; covalent: electrons shared between non-metals; one example of each; both lead to stable outer shells | `electron_transfer`, `electron_sharing`, `comparison_table` | "ionic bonds share electrons", "covalent bonds transfer electrons", "ionic bonds form between two non-metals" |

`chemistry_data.py` holds:

- `ELEMENTS`: H, C, N, O, F, Na, Mg, Cl, each with atomic number, name, shell configuration, group, metal/non-metal category and CPK colour.
- `MOLECULES`: H2, Cl2, F2, O2, N2, HCl, each with atoms and shared pairs.
- `IONIC_COMPOUNDS`: NaCl, NaF, MgO, each with donor, acceptor and electrons transferred.
- `SUBSTANCE_PH`: reference values with tolerance, e.g. lemon juice 2, vinegar 3, black coffee 5, pure water 7, blood 7.4, sea water 8, baking soda 9, ammonia 11.5, bleach 13. **Each value must be verified against a published reference (e.g. USGS "pH and Water") during implementation, with the source cited in the file.**

### 5.2 Topic resolution (API, synchronous, free)

1. Normalize the query: lowercase, strip punctuation, collapse whitespace.
2. Score each pack by its matched `all_of` groups, subject to its `any_of` / `none_of` rules.
3. The highest score wins; ties go to the pack with more groups.
4. No match → `422 UNSUPPORTED_TOPIC` with `details.supported_topics`.

`ionic_vs_covalent` requires both "ionic" and "covalent". `covalent_bonds` has `none_of: [ionic]`.
`ph_scale` matches "ph", "p h", "acidity", "alkalinity".

### 5.3 Storyboard schema (`schema_version = "1"`)

```python
class Storyboard(BaseModel):
    schema_version: Literal["1"]
    topic_id: str
    title: str                       # ≤ 60 chars
    learning_objective: str          # ≤ 160 chars
    scenes: list[Scene]              # 5..7

class Scene(BaseModel):
    on_screen_title: str             # ≤ 40 chars
    narration: str                   # 25..60 words (checked in code)
    visual: Visual                   # discriminated union on "type"
```

The visual catalogue is closed. The LLM selects; code resolves every number.

| `type` | LLM-provided fields | Code-resolved at render-plan time |
|---|---|---|
| `title_card` | `subtitle` ≤ 60 | learner's question (verbatim, sanitized) |
| `key_points` | `points[2..4]` ≤ 60 each | — |
| `ph_scale` | `markers[2..5]`: `SubstanceName` enum; `highlight`: acidic/neutral/basic/none | marker pH values, zone boundaries, colours |
| `log_ratio` | `from_ph`, `to_ph` (ints 0–14, \|Δ\| ∈ {1,2}) | ratio 10^\|Δ\|, dot-grid sizes |
| `atom_shells` | `elements[1..2]`: `ElementSymbol` enum | shell configuration, valence electrons |
| `electron_sharing` | `molecule`: `Molecule` enum | atoms, shared pairs |
| `electron_transfer` | `compound`: `IonicCompound` enum | donor, acceptor, electrons moved, ion charges |
| `comparison_table` | `left_header`/`right_header` ≤ 16; `rows[2..4]` of `aspect` ≤ 20, `left`/`right` ≤ 28 | — |

`scene.index` is assigned by code.

The Gemini call uses `response_mime_type="application/json"` with the JSON schema generated from
these models. If the API rejects the discriminated-union construct, the visual is flattened into one
object (`type` plus optional fields) for the provider schema, and G1 still validates the strict model.
This must be verified in the first live writer test.

### 5.4 Writer and critic

**Writer**
- Model `gemini-2.5-flash`, temperature 0.4, thinking budget 1024 tokens, max output 4096.
- Prompt `prompts/writer_v1.md` contains:
  - Audience: English-speaking secondary-school learners, ages 13–16.
  - The learner query wrapped as data in `<learner_query>…</learner_query>`, with the instruction that it is not an instruction.
  - Authoritative facts, misconceptions to avoid, the allowed visual catalogue with usage guidance, and the structure rules.
  - Style: friendly, concrete everyday examples, one idea per scene, no markdown.
- Rewrite prompt `rewrite_v1.md` = previous storyboard + the exact violation list from G1/G2/G3.

**Critic**
- Model `gemini-2.5-flash-lite`, temperature 0, prompt `critic_v1.md`.
- Input: learner query, authoritative facts, storyboard, and G2's `unverified_numeric_claims`.
- Output schema: `{accuracy, relevance, clarity, alignment: 1..5, issues[]: {scene_index|null, severity: blocker|major|minor, message}}`.
- Pass iff all four scores ≥ 4 and there is no `blocker`.

Prompt files carry a version that is recorded in checkpoints and manifests.

### 5.5 Gates

| Gate | Checks |
|---|---|
| **G1 Schema** | Pydantic strict validation: types, enums, lengths, counts. |
| **G2a Rules** | Narration 25–60 words per scene; estimated total duration (words ÷ 2.5 w/s + padding) 45–100 s; scene 0 is `title_card`; last scene is `key_points`; visuals ⊆ `allowed_visuals`; `required_visuals` present; no two scenes with identical visual spec; every `required_fact` detected; no misconception pattern matches. |
| **G2b Chemistry consistency** | `log_ratio` \|Δ\| ∈ {1,2}; `atom_shells` elements ⊆ pack `allowed_elements`; `electron_sharing` molecule ⊆ `allowed_molecules`; `electron_transfer` compound ⊆ `allowed_compounds`, and its donor is a metal and acceptor a non-metal (holds by construction of the enum, asserted anyway). |
| **G2c Claims in narration** | Checkers split narration into clauses (on `.;` and `while`, `whereas`, `but`) and verify: **ph_classification** (a pH number with acidic/neutral/basic word → class derived from the number); **ph_ratio** (N units with multiplier M → M == 10^N; "twice/double" per unit → violation); **substance_ph** (reference substance with a number → within tolerance); **ph_range** (scale bounds must be 0 and 14); **valence_claims** (element with a stated count of outer/valence electrons → matches `ELEMENTS`); **shared_pairs_claims** (molecule with stated pair count → matches `MOLECULES`). A clause with several numbers and several class words is **not guessed**: it is recorded as an unverified claim. Numbers matched by no checker are recorded in `unverified_numeric_claims` and forwarded to the critic. |
| **G2d Layout pre-flight** | For every LLM-provided string, look up its slot in `render/layout.py` (the same slot definitions the Manim templates use). Measure with Pillow `ImageFont.truetype(assets/fonts/Inter…, size)`, with greedy word wrapping and a 5 % safety margin. Check width ≤ slot width, height ≤ slot height, lines ≤ slot max lines, slot inside the 90 % title-safe area. Slots are non-overlapping by construction; render layout tests assert this. |
| **G3 Critic** | §5.4. Critic error or timeout → `critic_skipped=true`, gate passes on G1+G2. |

Gates run cheapest first. G3 runs only when G1 and G2 pass.

### 5.6 Loop and fallback

```
for attempt in 1..3:
    storyboard = writer(prompt if attempt == 1 else rewrite(prev, violations))
    violations = G1 ∪ G2 (stop early on G1 failure)
    if not violations: violations = G3   (or critic_skipped)
    if not violations: return (storyboard, generation_source="llm")
return (curated_storyboard(topic), generation_source="curated_fallback", fallback_reason=…)
```

`fallback_reason` is one of:

- `LLM_VALIDATION_FAILED` (with the last violation codes)
- `LLM_PROVIDER_ERROR` (non-retryable 4xx, or retries exhausted)
- `BUDGET_EXCEEDED_JOB`
- `BUDGET_EXCEEDED_DAILY`

The curated storyboard is validated by G1+G2 at start-up and in tests. If it fails at runtime, the job
fails with `CURATED_FALLBACK_INVALID` (non-retryable).

The reviewing-stage checkpoint stores the approved storyboard, `generation_source`, `fallback_reason`,
the gate reports for every attempt, and critic scores.

### 5.7 Input handling

- `query`: 3–300 characters, stripped of control characters.
- The query is shown on the title card through Manim `Text` (never `MarkupText`) to prevent Pango markup injection.
- Caption text escapes ASS override braces `{}` and backslashes.
- All LLM strings rendered on screen use `Text`. Subscripts in fixed labels (H₂O) come from code-owned markup only.

## 6. Checkpoints and resumability

- Scratch outputs are written under `work/<job_id>/attempt-<n>/<stage_key>/`. Directories are immutable once promoted.
- **Promotion** is one SQLite transaction:
  1. Fenced `UPDATE jobs SET updated_at=:now WHERE <fence>` (rowcount must be 1).
  2. `INSERT OR REPLACE INTO stage_checkpoints(job_id, stage_key, attempt, input_hash, manifest_sha256, versions_json, …)`.
  3. Commit.

  Then `stage.json` (file list with sha256, input hash, versions, attempt) sits in the attempt directory. The DB row is the authority for which attempt's output is current.
- A checkpoint is **valid** only if **all** of the following hold:
  - the DB row exists;
  - the directory and `stage.json` exist;
  - `sha256(stage.json)` equals `manifest_sha256`;
  - every listed file exists with a matching sha256;
  - `input_hash` equals the hash of the current stage input;
  - `versions_json` matches the current `pipeline_version`, `schema_version`, prompt versions, `topic_pack_version`, `renderer_version` and `voice` config.
- On claim, the orchestrator walks the stages and reuses each valid checkpoint (event `stage_skipped_checkpoint`). The first invalid or missing one is recomputed.
- Stage keys: `reviewing`, `voicing`, `voicing/scene-NN`, `rendering/scene-NN`, `rendering`, `quality_check`.
- **Retention:** work directories of succeeded jobs are deleted after publication. Failed jobs keep theirs (debugging and resume).

## 7. Narration audio (Part 3 — voicing)

### 7.1 Providers

```python
class TTSProvider(Protocol):
    name: str
    def synthesize(self, text: str, voice: VoiceSpec) -> SynthesisResult: ...
    # SynthesisResult: wav_bytes (PCM s16le mono), sample_rate, duration_s, usage, cost_entry

class ASRProvider(Protocol):
    name: str
    def transcribe(self, wav_bytes: bytes, sample_rate: int) -> Transcript: ...
```

| Provider | Role | Details |
|---|---|---|
| `GeminiTTSProvider` | primary | `gemini-2.5-flash-preview-tts`, voice `Kore`, style prefix from `voicing` config (versioned), PCM 24 kHz s16le mono; cost = actual from `usage_metadata` |
| `EdgeTTSProvider` | fallback (different vendor) | voice `en-US-AriaNeural`, MP3 decoded to PCM via ffmpeg; `cost_source=free` |
| `GeminiASRProvider` | A5 verification | `gemini-2.5-flash-lite`, temperature 0, verbatim transcription prompt; cost actual |
| `FakeTTSProvider` / `FakeASRProvider` | tests | deterministic tones/transcripts with scriptable failures |

### 7.2 Per-scene synthesis and voice consistency

- Synthesis runs per scene on `spoken_text` = narration with the pack's pronunciations applied.
- Concurrency limit is 2.
- **TTS cache key:** `sha256(provider, model, voice, style_version, spoken_text)`, stored under `data/cache/tts/`. Cached clips are re-validated (A1–A4) before reuse.
- **One voice per video.** A scene that exhausts retries on the current provider triggers a **video-level switch**: all scenes are re-voiced on the fallback provider, cache hits included. The switch emits a `voice_provider_switched` event; the cost of abandoned calls stays in the ledger.
- **Budget pre-check before voicing:** projected cost = Σ expected seconds × 25 tokens/s × price. If it exceeds the remaining per-job or daily budget, the whole video uses the fallback provider from the start (`voice_fallback_reason=BUDGET_EXCEEDED_*`).

### 7.3 Clip validation

| Check | Rule (defaults are settings) |
|---|---|
| **A1** | WAV decodes; mono; expected sample rate; ≥ 0.5 s of samples. |
| **A2** | `duration / (words / 2.5)` within [0.6, 1.6], measured after trimming. |
| **A3** | Leading/trailing silence below −40 dBFS trimmed to ≤ 0.1 s; RMS of trimmed clip > −35 dBFS. |
| **A4** | No internal silence (−40 dBFS) longer than 1.5 s (`silencedetect`). |
| **A5** | Reverse ASR: normalize both texts (lowercase, strip punctuation, number words → digits, join spelled letters "p h" → "ph"). Word error rate is computed with a Levenshtein distance in code. The threshold is **set from `evals/calibrate_gates.py` output** and recorded with a link to the calibration report. Until a calibration report exists, A5 runs in **report-only mode** (WER recorded, no gating). ASR unavailable → `asr_skipped=true`. |

A1–A4 are the media validity floor; A5 is the semantic layer.

**Retry policy per clip:**
1. Up to 2 re-syntheses on the current provider.
2. Then the video-level provider switch.
3. Then `TTS_VALIDATION_FAILED`, or `TTS_PROVIDER_UNAVAILABLE` if both providers error.

A video is never produced without narration.

**Voicing checkpoint:** per-scene clips, durations, provider, voice, check results and WER values.

## 8. Rendering, assembly and media QA (Part 3)

### 8.1 Render plan and renderer boundary

```python
class Renderer(Protocol):
    name: str
    version: str
    def render(self, plan: RenderPlan, out_dir: Path, scenes: list[int] | None = None) -> RenderResult: ...
    # returns silent per-scene MP4s + per-scene result records
```

`RenderPlan` is renderer-agnostic.

- **Frame timeline.** For each scene:
  `duration_s = lead_in (0.3) + clip_duration + tail (0.7; last scene 1.5)`.
  Cumulative boundaries: `start_frame = round(cum_start × fps)`, `end_frame = round(cum_end × fps)`. Rounding never accumulates, so `Σ frames = round(total × fps)`.
- **Resolved visuals.** Every number comes from `chemistry_data` (§5.3 table).
- **Beats.** Each marker, row, point or electron event gets `at_s = lead_in + (char_offset_of_first_mention / len(narration)) × clip_duration`. Beats are clamped to `[lead_in, lead_in + clip_duration − 0.5]` and made monotonic. Items never mentioned are spread evenly over the remaining time.
- Also carried: `on_screen_title`, the question text, theme id, `1280×720`, `fps=30`.

`FakeRenderer` produces solid-colour or `testsrc` MP4s with the exact frame counts (ffmpeg) for tests.

### 8.2 ManimRenderer

- Pinned `manim==0.21.0`.
- Invocation: `python -m app.render.manim_runner --plan plan.json --out <dir> [--scenes 0,3]`, as **one subprocess for all requested scenes**.
- The process starts in its own group (`start_new_session` on POSIX, `CREATE_NEW_PROCESS_GROUP` on Windows). On timeout (default 15 min per job) or lease loss, the psutil process tree is killed.
- The runner registers the bundled font with `manimpango.register_font` and builds one Scene per planned scene from `render/manim/templates/<type>.py`. Each scene renders to `scene-NN.mp4` and writes `scene-NN.result.json` with `{ok, frames, error_type, message, traceback_tail}`.
- **Classification:**
  - Template exception or layout assertion → `RENDER_TEMPLATE_ERROR` / `RENDER_LAYOUT_ERROR` (non-retryable, no re-render).
  - Missing result file or non-zero runner exit → crash (`RENDER_FAILED`, retryable).
  - Crashed scenes are re-rendered once in a new subprocess with `--scenes` limited to them.
- **Frame exactness.** Each template's final wait fills the scene to `end_frame − start_frame`. After rendering, `ffprobe -count_frames` must match. A difference of ≤ 1 frame is corrected in assembly (trim or clone last frame) and logged; a larger difference is `RENDER_TEMPLATE_ERROR`.
- Successful scenes are checkpointed individually (`rendering/scene-NN`).
- **Templates only extend holds.** They never speed animations up. Every element animates in; the scene fades out in its last 0.3 s.
- **Theme:**
  - Background `#0F172A`; text contrast ≥ 4.5:1.
  - Discrete universal-indicator palette for pH 0–14; electrons yellow; CPK element colours.
  - Layout slots: title 0–15 %, visual 15–78 %, captions 80–95 % of height; scene progress dots.

| Template | Motion |
|---|---|
| `title_card` | title writes in, learner's question types in, subtle drifting molecule glyphs |
| `ph_scale` | 15 segments stagger in with numbers; zone labels; markers drop at beats with resolved pH; highlight pulse |
| `log_ratio` | two H⁺ dot grids (1 vs 10, or 1 vs 100 as 10×10) with "10× / 100× more H⁺" |
| `atom_shells` | Bohr model; valence electrons highlighted and slowly orbiting |
| `electron_sharing` | atoms approach; valence electrons move into overlap; shells overlap; "full shell ✓" counters |
| `electron_transfer` | electron(s) arc donor → acceptor; charges appear; attraction arrows |
| `comparison_table` | headers, then rows reveal at beats with transfer/share icons |
| `key_points` | ticked points at beats; question recap line |

### 8.3 Captions

- **Cue building.** Display narration (not spoken text) is split per sentence; long sentences split at commas. Each cue is ≤ 2 lines × 42 characters, wrapped with the same font metrics.
- **Timing.** Cue time is proportional to character count inside the scene's clip window. Scene boundaries are exact; timing within a scene is approximate (documented).
- **Output.** `captions.vtt` for players, plus an ASS file for burn-in.

### 8.4 MediaAssembler (ffmpeg CLI)

1. **Audio.** Decode each clip to float32 48 kHz mono. Apply per-clip `loudnorm` (I = −16 LUFS, TP = −1.5 dB, LRA = 11). Place clips at exact sample offsets `(start_frame / fps + lead_in) × 48000` in one numpy buffer of `total_frames / fps × 48000` samples, then write `narration.wav`.
2. **Video.** Concatenate scene MP4s with the concat demuxer (stream copy).
3. **Final encode.** Burn ASS captions (`subtitles` filter, `fontsdir=assets/fonts`, relative paths with `cwd` set to the work directory to avoid Windows drive-colon escaping). H.264 High, `yuv420p`, CRF 23, preset `medium`, 30 fps; AAC 128 kbps 48 kHz; `-movflags +faststart`.
4. **Images.** `thumbnail.jpg` from the title card midpoint. `contact_sheet.jpg` from each scene's midpoint, in a grid.

- Every ffmpeg call goes through `render/ffmpeg.py` with a timeout and a captured stderr tail.
- A non-zero exit with I/O or unknown stderr → `ASSEMBLY_FAILED / ffmpeg_transient`, re-run once. A missing or invalid input file → `ASSEMBLY_FAILED / invalid_input`.

### 8.5 Media QA (deterministic)

| Check | Rule |
|---|---|
| **Q1 Container** | exactly one h264 `yuv420p` 1280×720 30 fps stream and one AAC 48 kHz stream; `moov` before `mdat` |
| **Q2 Duration** | total 40–120 s; \|video − audio\| ≤ 0.1 s; \|actual − planned\| ≤ 0.5 s |
| **Q3 Audio** | `volumedetect` mean > −35 dB and max ≤ −1.0 dB; `ebur128` integrated loudness in [−19, −13] LUFS; no `silencedetect` gap > 2.0 s outside planned padding |
| **Q4 Video** | `blackdetect` (pic_th 0.98) no segment ≥ 1.0 s; `freezedetect` no freeze ≥ 10 s |
| **Q5 Sanity** | per-scene midpoint frame luma standard deviation > 8; perceptual-hash Hamming distance between consecutive scene frames ≥ 6. These are sanity checks against blank or duplicated renders, not a semantic quality evaluation. |

- The Q3–Q5 thresholds are confirmed against renders of the three curated storyboards during implementation. Any change is recorded in `docs/RELIABILITY.md`.
- **Failure handling.** File read or ffprobe I/O errors → `QUALITY_CHECK_FAILED / io_error` (retryable). A deterministic rule violation → `QUALITY_CHECK_FAILED / deterministic_output_bug` (non-retryable). The report is always written.
- No VLM inspects individual videos. Semantic correctness is owned by topic facts, G2 and the critic; template visual quality is owned by build-time layout and golden tests (§10.1).

## 9. Artifacts, API, cost, observability

### 9.1 ArtifactStore

```python
class ArtifactStore(Protocol):
    def publish(self, job_id: str, files: dict[str, Path], manifest: Manifest) -> list[ArtifactRef]: ...
    def open(self, job_id: str, name: str) -> Path: ...
    def verify(self, job_id: str) -> Manifest | None: ...     # manifest present and all hashes match
    def url_for(self, job_id: str, name: str) -> str: ...
```

**Local implementation (`data/artifacts/<job_id>/`):**

- Each file is copied to `<name>.tmp-<attempt>` and then `os.replace`d. `manifest.json` is written last, the same way.
- A valid manifest (present, every hash matches) means publication is complete. After publishing, the worker performs the fenced update to `succeeded`.
- **Recovery.** If a claimed job has a manifest that `verify()` accepts and whose `pipeline_version` and `job_id` match, it goes straight to `succeeded` (event `publish_recovered`).

**Published files:**

| File | Contents |
|---|---|
| `video.mp4` | the finished video |
| `thumbnail.jpg` | title card midpoint |
| `captions.vtt` | sidecar captions |
| `storyboard.json` | the approved storyboard |
| `quality_report.json` | all G/A/Q gate results, attempts, critic scores, WER |
| `contact_sheet.jpg` | one frame per scene |
| `manifest.json` | file list with sha256 and bytes; `pipeline_version`, `schema_version`, prompt versions, models, voice/provider, `renderer_version`, `topic_pack_version`, `generation_source`, `fallback_reason`, cost summary, per-stage durations |

Production mapping: GCS bucket with object generation preconditions; `url_for` returns signed URLs and `/content` responds with 302.

### 9.2 HTTP API (`/v1`)

All errors use the envelope `{"error": {"code", "message", "retryable", "details"}}`.

| Method & path | Success | Errors |
|---|---|---|
| `POST /v1/videos` body `{"query": str}` + optional `Idempotency-Key` | `202` `VideoJob` + `Location` | `422 INVALID_REQUEST`, `422 UNSUPPORTED_TOPIC`, `409 IDEMPOTENCY_KEY_REUSED` |
| `GET /v1/videos?status=&topic_id=&limit=20(≤100)&cursor=` | `200 {items: [VideoJob], next_cursor}` newest first, keyset cursor on `(created_at, id)` | `422 INVALID_REQUEST` |
| `GET /v1/videos/{id}` | `200 VideoJobDetail` | `404 NOT_FOUND` |
| `GET /v1/videos/{id}/content` | `200 video/mp4` (Range supported) | `404`, `409 VIDEO_NOT_READY {status, stage}` |
| `GET /v1/videos/{id}/artifacts` | `200 [ArtifactRef]` | `404`, `409 VIDEO_NOT_READY` |
| `GET /v1/videos/{id}/artifacts/{name}` | file; `name` from the published allow-list only | `404`, `409` |
| `POST /v1/videos/{id}/retry` | `202 VideoJob` | `404`, `409 NOT_RETRYABLE` |
| `GET /v1/topics` | `200 [{topic_id, title, canonical_query}]` | — |
| `GET /healthz` | `200 {db, ffmpeg, worker_heartbeat_age_s}` | `503` if DB or ffmpeg unavailable |

**`VideoJob` fields:** `id`, `query`, `topic_id`, `status`, `stage`, `progress` (0–1, stage-based), `attempt`, `generation_source`, `fallback_reason`, `error`, `created_at`, `updated_at`, `started_at`, `finished_at`, `links {self, content, artifacts}`.

**`VideoJobDetail` adds:** `events[]`, `quality` summary (gate outcomes, attempts, critic scores, `critic_skipped`, `asr_skipped`, `voice_provider`), `cost {total_usd, actual_usd, estimated_usd, free_calls, by_operation, pricing_version}`, `artifacts[]`, `stage_durations_s`.

**Process model.** The worker runs embedded in the API process by default (`RUN_EMBEDDED_WORKER=true`, one worker thread, pipeline is synchronous). It can also run as `python -m app.worker`. Any number of worker processes may share the DB.

### 9.3 Cost ledger, pricing, budgets

- **Every paid or free provider call writes a `cost_entries` row, including failed calls and retries.**
  - `cost_source=actual`: computed from `usage_metadata` × price.
  - `cost_source=estimated`: usage unavailable. Text tokens ≈ characters ÷ 4; TTS audio tokens = seconds × 25; audio input tokens = seconds × 32.
  - `cost_source=free`: edge-tts.
  - `render_compute` is always estimated: measured CPU seconds × configured vCPU × Cloud Run price (a production estimate).
- **`cost/pricing.py`:** `PRICING_VERSION = "2026-09-13"`. Each price records `input_per_m`, `output_per_m` (or `per_second`), `source_url` and `verified_on`. Current values from the Gemini pricing page:

  | Model | Input / 1M | Output / 1M |
  |---|---|---|
  | 2.5 Flash | $0.30 | $2.50 |
  | 2.5 Flash-Lite | $0.10 text, $0.30 audio | $0.40 |
  | 2.5 Flash TTS | $0.50 text | $10.00 audio |

  Cloud Run instance-based prices ($0.000018 per vCPU-s, $0.000002 per GiB-s) come from a secondary source and **must be re-verified on the official page before publication in `docs/COST.md`**.
- **Budgets:** `COST_JOB_BUDGET_USD=0.10`, `COST_DAILY_BUDGET_USD=1.00` (UTC day, sum over `cost_entries`). Before each paid call, `spent + projected(call) > budget` triggers the degrade path:
  - writer/critic → curated fallback;
  - TTS → fallback provider for the whole video (§7.2);
  - ASR → `asr_skipped` with reason `budget`.

  Degradation emits `budget_degraded` and never fails the job by itself.

**Expected per-video cost** (75 s video, to be replaced by measured ledger values):

| Item | Estimate |
|---|---|
| writer attempt | ≈ $0.007 |
| critic | ≈ $0.0005 |
| TTS | ≈ $0.019 |
| ASR | ≈ $0.0008 |
| compute | ≈ $0.005 |
| **typical** | **≈ $0.03** |
| worst case (3 writer attempts + TTS retries) | ≈ $0.055 |

TTS is the dominant cost (~60 %).

**Documented, not implemented:** reuse of canonical-query videos across learners (a product decision that trades against personalization); Batch API for TTS (−50 %, latency-tolerant); Spot render workers.

### 9.4 Observability

- **Logs.** JSON lines on stdout via stdlib `logging`, with contextvars `job_id`, `attempt`, `stage`, `worker_id`.
- **Event types:**
  - Job and lease: `job_created`, `job_claimed`, `lease_reclaimed`, `lease_lost`, `retry_scheduled`, `retry_requested`, `job_succeeded`, `job_failed`.
  - Stages and checkpoints: `stage_started`, `stage_completed`, `stage_skipped_checkpoint`.
  - Content: `writer_attempt`, `gate_result`, `critic_skipped`, `fallback_used`.
  - Providers and budget: `provider_retry`, `budget_degraded`.
  - Voicing: `tts_clip_retry`, `voice_provider_switched`, `asr_skipped`.
  - Rendering and QA: `render_scene_failed`, `render_scene_retried`, `qa_result`.
  - Publication: `artifact_published`, `publish_recovered`.
- Event `data_json` stays small (codes, counts, scores, durations). Large payloads live in checkpoints.

## 10. Testing and evaluation (Part 4)

### 10.1 Test suites

**Markers:** `unit` (default), `integration`, `render`, `live`. **CI** runs ruff, mypy, unit and integration (ffmpeg installed, no API keys).

**Unit** (target < 30 s, no network, no Manim)

*Jobs and API (Part 1)*
- **Transition table:** every legal and illegal state pair.
- **Atomic claim:** 8 threads with separate connections race over 50 jobs; each job is claimed exactly once.
- **Lease:**
  - With a `FakeClock`, an expired lease is reclaimed by worker B with `attempt+1`.
  - A heartbeat prevents the reclaim.
  - A stale worker's write → `StaleLeaseError` and 0 rows changed.
  - `lease_claims ≥ max` → the sweep produces `WORKER_LOST`.
- **Stateful property test (Hypothesis):** random sequences of claim, heartbeat, crash, clock advance, stage failure and retry. Invariants: never two unexpired leases on one job; once the clock passes every lease and a live worker keeps polling, no job stays in `running` and every job eventually becomes terminal.
- **Retry:** two concurrent retries → exactly one wins. Scheduled retry honours `not_before` and the count limit.
- **Idempotency:** replay returns the same job; a different body → 409; concurrent same-key inserts → one job.
- **API contract:** `TestClient` with fakes checks status codes, `Location`, the error envelope, pagination cursors, the artifact name allow-list and `/healthz`.
- **Observability:** JSON log records carry `job_id` and `attempt`; the event timeline is ordered.

*Content (Part 2)*
- **Resolver:** the three canonical queries plus paraphrases; the covalent vs ionic-vs-covalent precedence; unsupported queries.
- **G1:** a valid and an invalid fixture per constraint.
- **G2a–b:** each rule has a passing and a failing fixture.
- **G2c claims:** table-driven:
  - `"A pH of 7 is neutral"` ✓
  - `"pH 7 is acidic"` ✗
  - `"two pH units means 100 times"` ✓
  - `"two pH units means 20 times"` ✗
  - `"each unit is twice as acidic"` ✗
  - `"lemon juice has a pH of 9"` ✗
  - `"sodium has one outer electron"` ✓
  - `"chlorine has two outer electrons"` ✗
  - an ambiguous multi-number clause → recorded unverified, not failed
- **G2d layout:** worst-case strings with the real font (fit / too wide / too many lines).
- **All three curated storyboards pass G1 and G2.**
- **Writer loop with scripted `FakeLLM`:**
  - invalid → valid gives `llm`, and the rewrite prompt contains the exact violation messages;
  - three invalid → `curated_fallback` with `LLM_VALIDATION_FAILED`;
  - 401 → immediate fallback with no retry;
  - 429 twice → backoff (fake sleeper) → success;
  - critic error → `critic_skipped`;
  - critic < 4 → rewrite;
  - budget exceeded mid-loop → no further LLM calls, `BUDGET_EXCEEDED_JOB`.

*Media (Part 3)*
- **Audio checks** on numpy-synthesised WAVs: A1–A4 pass and fail cases, trimming; A5 normalization and WER edge cases; `asr_skipped`; report-only mode.
- **Voicing policy:** a scene failing after retries → the whole video is re-voiced on the fallback; valid cached clips are not re-synthesised (fake call counts); the ledger includes failed calls.
- **Budget pre-check** routes the whole video to the fallback voice.
- **Timeline:** frame totals exact across scenes; beats within bounds and monotonic.
- **Captions:** line limits, VTT validity, monotonic timing, ASS escaping.
- **Checkpoint validity:** one test per invalidation condition (missing row, missing file, `stage.json` hash, file hash, input hash, version mismatch). Promotion with a stale fence is rejected.
- **Publication:** crash after files but before the DB update → recovery to `succeeded`; tampered file hash → not treated as published.
- **Error classification table:** every `(code, cause)` → `retryable`.
- **Cost:** actual vs estimated vs free; pricing version stored; job and daily budget arithmetic.
- **Input sanitization:** Pango markup in the query rendered literally; ASS braces escaped.

**Integration** (`-m integration`, real ffmpeg, SQLite file, worker loop; fake LLM/TTS/ASR and `FakeRenderer`)
- One job runs end to end to `succeeded`, with a verified manifest and all published files.
- **Injected faults:**
  - black video → Q4 `deterministic_output_bug`;
  - audio/video duration mismatch → Q2;
  - ffmpeg transient failure → assembly re-run succeeds;
  - render crash → scene re-render;
  - worker killed mid-job (real subprocess, 3 s lease) → another worker resumes from checkpoints without re-calling the fake LLM.

**Render** (`-m render`, Manim)
- **Layout tests:** each template with worst-case fixture strings, mobjects built without rendering; everything inside the title-safe area; no text bounding-box overlaps; every G2d-accepted string fits without shrinking below 0.9×.
- **Golden snapshots:** the last frame of each template at low quality is compared by perceptual hash (tolerance) with PNGs committed under `tests/golden/`, generated inside the Docker image so CI and container match.
- **Full render:** the `ph_scale` curated storyboard renders end to end with real assembly and passes media QA.

**Live** (`-m live`, requires `RUN_LIVE=1` and a key; each test spends < $0.02)
- One writer call is schema-valid (verifies the §5.3 structured-output assumption).
- One Gemini TTS clip passes A1–A4.
- One ASR transcription.
- One edge-tts clip.

### 10.2 Gate calibration (`evals/calibrate_gates.py`)

- **A5.** Voice every narration line of the three curated storyboards → good set. Build a bad set from truncated clips, clips synthesised with the style instruction included in the text, and clips with substituted chemistry words. Report both WER distributions and pick a threshold that separates them. The chosen value is written to `evals/results/calibration-<date>.md` and referenced from config.
- **Critic.** Run on the curated storyboards (expected pass) and on seeded-defect storyboards with labels (wrong pH classification, missing logarithmic fact, visual/narration mismatch, off-topic scene). Report detection rate and false-rejection rate.

### 10.3 Reliability evaluation (`evals/reliability_eval.py`)

**`--mode chaos`** (free)
- Fake providers with injected rates: LLM invalid output 50 %, LLM 429 20 %, TTS truncated clip 30 %, render crash 10 %; worker process killed mid-job 10 % with a 3 s lease.
- Runs 50 jobs through the real API and worker processes.
- **Invariants:**
  1. 100 % of jobs terminal.
  2. 0 jobs stuck in `running`.
  3. 0 published artifacts failing `verify()` or media QA.
  4. Fallback is used exactly when the writer loop exhausts.
  5. Ledger rows exist for failed calls.

**`--mode live`**
- The three canonical queries × `--runs` (default 5) go through the HTTP API.
- A pre-flight projection times the p95 per-video estimate. The run refuses to start above `--max-usd` (default $1.00) or when cumulative ledger spend would exceed the $3.00 development cap.
- **Report** (`evals/results/<timestamp>/report.md` + `runs.jsonl` + one contact-sheet grid per topic):
  - success rate; `llm` vs `curated_fallback`; writer attempt distribution; G2 violation histogram;
  - critic score mean/min and `critic_skipped`;
  - clip retry rate, provider switches, WER distribution, `asr_skipped`;
  - render/QA failures by cause;
  - video duration mean/min/max/coefficient of variation;
  - cost per video mean/p95/max, with the actual/estimated share;
  - stage p50/p95 wall time;
  - per-run required-fact coverage and visual-type sequence.
- Numbers are reported as measured, including shortfalls, with analysis.

`scripts/export_samples.py` picks, per topic, the best `succeeded` + `llm` run (fewest writer attempts, then highest minimum critic score, then lowest WER) and copies its files to `samples/<topic_id>/`. The README states the samples are best-of-N and links the full report.

## 11. Deliverables

| Challenge deliverable | Location | Owner |
|---|---|---|
| FastAPI codebase | `app/` | implementation |
| README (setup, run, API, tests) | `README.md`, commands executed from a clean clone | implementation |
| Architecture note | `docs/ARCHITECTURE.md`: job lifecycle, persistence/artifact boundary, AI/video-generation boundary | implementation |
| Demo / API walkthrough | `scripts/demo.py` + `docs/API_WALKTHROUGH.md` with real captured output; screen recording by the candidate | implementation + candidate |
| Three best videos + input queries | `samples/` | implementation |
| GitHub link, read access for careers@, praveen.k@, wayne.le@growtrics.ai | — | candidate |
| Zip | `scripts/make_zip.sh` (`git archive`, tracked files only) | implementation writes, candidate runs |
| Recording (screen + face) on Google Drive | — | candidate |

**Supporting documents:**
- `docs/RELIABILITY.md`: failure-mode table and eval results.
- `docs/COST.md`: measured costs and cited prices.
- `docs/DECISIONS.md`: decision, alternatives, rationale.
- `docs/AI_WORKFLOW.md`: how Claude Code was steered, **with real events only**, e.g. the spike that changed font and palette decisions, cross-review rounds, and correcting the premise that code existed before it did.

**Interview preparation** documents (Vietnamese) live **outside the repository** in `D:\growtrics-interview-prep\` and are never committed or zipped. Each file ends with a *Trả lời phỏng vấn* section.

| File | Contents |
|---|---|
| `00_TONG_QUAN` | pitch; JD ↔ project map |
| `01_KIEN_TRUC_JOB` | job architecture |
| `02_PIPELINE_NOI_DUNG` | content pipeline |
| `03_GIONG_RENDER_QA` | voice, render, QA |
| `04_CHI_PHI` | cost |
| `05_TEST_DANH_GIA` | tests and evaluation |
| `06_LAM_VIEC_VOI_AGENT` | working with the coding agent |
| `07_CAU_HOI_PHONG_VAN` | 50+ questions with English answers and Vietnamese notes |
| `08_DIEN_TAP_DEMO` | demo script and fallback plan |

## 12. Implementation workflow

1. Plan with phases:
   - P0 scaffold/CI
   - P1 domain + repository + worker
   - P2 API + idempotency
   - P3 topic packs + schema + G1/G2
   - P4 providers + cost ledger + budgets
   - P5 writer/critic loop + fallback
   - P6 voicing + audio checks + cache
   - P7 render plan + templates + ManimRenderer
   - P8 assembler + captions + QA + checkpoints + publication
   - P9 integration + chaos eval
   - P10 calibration + live eval + samples
   - P11 submission docs + Docker + zip
   - P12 interview prep docs
2. Every phase: tests first, implement, run, then commit. Commit messages are in English, use multiple `-m` paragraphs and explain *why*.
3. Code review checkpoints after P2, P5, P8 and P10.
4. Pushing to GitHub, granting access, sending email and recording are done by the candidate.

**Stack:**
- **Runtime:** Python 3.12, uv, FastAPI, uvicorn, Pydantic v2, pydantic-settings, google-genai, edge-tts, manim 0.21.0, Pillow, numpy, psutil, PyYAML, httpx.
- **Dev:** pytest, hypothesis, ruff, mypy.
- **System:** ffmpeg ≥ 6 with libass.
- **Docker:** `python:3.12-slim` plus ffmpeg, cairo and pango runtime libraries.

## 13. Out of scope (with rationale in `docs/DECISIONS.md`)

Authentication and multi-tenancy; per-user rate limiting; human review queue (extension point:
a `needs_review` publication policy); vertical 9:16 layouts; non-English narration; topics beyond
the three packs; GCS/Firestore/Cloud Tasks adapters (mapping documented only); Prometheus;
job cancellation; idempotency-key expiry.

## 14. Refinements introduced while writing this spec (need confirmation)

1. **Scheduled automatic retry** for retryable stage failures (`running → queued` with `not_before`, at most 2, delays 60 s / 300 s) before a job becomes `failed`.
2. **`lease_claims`** counter separate from the monotonic `attempt` fencing token, so a manual retry resets the automatic-reclaim budget without reusing fencing tokens.
3. **`stage_checkpoints` table** as the DB authority for which immutable per-attempt checkpoint directory is current. This implements decision "only the lease holder may publish a checkpoint" without directory-rename races on Windows.
4. **`electron_sharing` / `electron_transfer` take a molecule or compound enum** instead of LLM-supplied atoms and pair counts. This is a stricter application of "code populates controlled numeric values".
5. **Input sanitization:** `Text` not `MarkupText` for user and LLM strings; ASS brace escaping.
6. **Render compute cost** is included in the per-video ledger as an estimated entry.
7. **A5 runs report-only until a calibration report exists**, so no threshold is invented.
