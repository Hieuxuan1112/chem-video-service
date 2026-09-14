"""Print the project spend state and unsettled reservations: `uv run python -m app.cost.report`."""

import json

from app.config import Settings
from app.cost.budget import SpendGuard
from app.cost.ledger import Ledger
from app.jobs.clock import SystemClock


def main() -> None:
    settings = Settings()
    clock = SystemClock()
    ledger = Ledger(settings.ledger_path(), clock)
    guard = SpendGuard(ledger, settings, clock)
    report = {
        "ledger_path": str(settings.ledger_path()),
        "is_project_ledger": settings.is_project_ledger(),
        # billable_usd is cumulative: every settled billable call ever recorded in this ledger,
        # across all runs, actual and estimated costs together. It is the figure the caps use.
        "scope": "project: all runs recorded in this ledger (not only the latest run)",
        **guard.state(),
        "calls": ledger.call_counts(),
        "breakdown": ledger.project_cost_summary(),
        "unsettled": ledger.unsettled(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
