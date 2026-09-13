import itertools

import pytest

from app.domain.errors import IllegalTransition
from app.domain.job import (
    ALLOWED_TRANSITIONS,
    STAGE_ORDER,
    JobStatus,
    Stage,
    check_transition,
    new_job_id,
    stage_progress,
)

LEGAL = {
    (JobStatus.QUEUED, JobStatus.RUNNING),
    (JobStatus.RUNNING, JobStatus.RUNNING),
    (JobStatus.RUNNING, JobStatus.SUCCEEDED),
    (JobStatus.RUNNING, JobStatus.FAILED),
    (JobStatus.RUNNING, JobStatus.QUEUED),
    (JobStatus.FAILED, JobStatus.QUEUED),
}


def test_allowed_table_matches_spec() -> None:
    assert set(ALLOWED_TRANSITIONS) == LEGAL


@pytest.mark.parametrize(("src", "dst"), list(itertools.product(JobStatus, JobStatus)))
def test_all_pairs(src: JobStatus, dst: JobStatus) -> None:
    if (src, dst) in LEGAL:
        check_transition(src, dst)
    else:
        with pytest.raises(IllegalTransition):
            check_transition(src, dst)


def test_progress_values() -> None:
    assert STAGE_ORDER == (
        Stage.SCRIPTING,
        Stage.REVIEWING,
        Stage.VOICING,
        Stage.RENDERING,
        Stage.QUALITY_CHECK,
        Stage.PUBLISHING,
    )
    assert stage_progress(JobStatus.QUEUED, None) == 0.0
    assert stage_progress(JobStatus.SUCCEEDED, Stage.PUBLISHING) == 1.0
    assert stage_progress(JobStatus.RUNNING, None) == 0.0
    assert stage_progress(JobStatus.RUNNING, Stage.SCRIPTING) == 0.0
    assert stage_progress(JobStatus.RUNNING, Stage.VOICING) == pytest.approx(2 / 6)
    assert stage_progress(JobStatus.RUNNING, Stage.PUBLISHING) == pytest.approx(5 / 6)
    values = [stage_progress(JobStatus.RUNNING, s) for s in STAGE_ORDER]
    assert values == sorted(values)
    assert all(0.0 <= v < 1.0 for v in values)


def test_job_id_prefix_and_uniqueness() -> None:
    ids = {new_job_id() for _ in range(1000)}
    assert len(ids) == 1000
    for job_id in ids:
        assert job_id.startswith("vid_")
        assert len(job_id) == len("vid_") + 32
