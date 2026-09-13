"""Service settings. Every threshold named in the spec is a field here."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Paths
    database_path: Path = Path("data/app.sqlite")
    ledger_db_path: Path | None = None  # None -> database_path
    project_ledger_path: Path = Path("data/app.sqlite")
    artifacts_dir: Path = Path("data/artifacts")
    work_dir: Path = Path("data/work")
    cache_dir: Path = Path("data/cache")
    fonts_dir: Path = Path("assets/fonts")

    google_api_key: SecretStr | None = None
    pipeline_version: str = "1.0.0"

    # Jobs and leases (spec §4.3, §4.6)
    lease_s: float = 60.0
    heartbeat_interval_s: float = 15.0
    max_claims: int = 3
    poll_interval_s: float = 1.0
    scheduled_retry_delays_s: tuple[float, ...] = (60.0, 300.0)
    run_embedded_worker: bool = True

    # Budgets (spec §9.3)
    warning_budget_usd: float = 3.00
    hard_project_cap_usd: float = 5.00
    cost_job_budget_usd: float = 0.20
    cost_daily_budget_usd: float | None = None

    # Provider calls and retries
    provider_timeout_s: float = 60.0
    retry_max_attempts: int = 3
    retry_base_s: float = 1.0
    retry_factor: float = 2.0
    retry_cap_s: float = 20.0

    # Writer and critic (spec §5.4)
    writer_model: str = "gemini-2.5-flash"
    writer_temperature: float = 0.4
    writer_thinking_budget: int = 1024
    writer_max_output_tokens: int = 4096
    max_storyboard_attempts: int = 3
    critic_model: str = "gemini-2.5-flash-lite"
    critic_max_output_tokens: int = 1024
    critic_min_score: int = 4

    # Voicing (spec §7)
    tts_model: str = "gemini-2.5-flash-preview-tts"
    tts_voice: str = "Kore"
    tts_style_version: str = "teacher-v1"
    edge_voice: str = "en-US-AriaNeural"
    tts_concurrency: int = 2
    clip_resynth_attempts: int = 2
    asr_model: str = "gemini-2.5-flash-lite"
    a5_wer_threshold: float | None = None
    words_per_second: float = 2.5
    a2_min_ratio: float = 0.6
    a2_max_ratio: float = 1.6
    silence_threshold_dbfs: float = -40.0
    trim_keep_s: float = 0.1
    a3_min_rms_dbfs: float = -35.0
    a4_max_internal_silence_s: float = 1.5

    # Rendering (spec §8)
    width: int = 1280
    height: int = 720
    fps: int = 30
    lead_in_s: float = 0.3
    tail_s: float = 0.7
    final_tail_s: float = 1.5
    render_timeout_s: float = 900.0

    # Storyboard rules (spec §5.3, §5.5)
    min_scenes: int = 5
    max_scenes: int = 7
    min_words: int = 25
    max_words: int = 60
    min_total_s: float = 45.0
    max_total_s: float = 100.0

    def ledger_path(self) -> Path:
        return self.ledger_db_path if self.ledger_db_path is not None else self.database_path

    def is_project_ledger(self) -> bool:
        return self.ledger_path().resolve() == self.project_ledger_path.resolve()
