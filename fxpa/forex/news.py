"""News blackout: no new entries from N minutes before to M minutes after high-impact events."""

from __future__ import annotations

import pandas as pd

from fxpa.config import Instrument


class NewsFilter:
    def __init__(self, events: pd.DataFrame | None, cfg: dict):
        p = cfg["news"]
        self.enabled = bool(p.get("enabled", True))
        self.before = pd.Timedelta(minutes=int(p["before_minutes"]))
        self.after = pd.Timedelta(minutes=int(p["after_minutes"]))
        impacts = {s.lower() for s in p.get("impacts", ["High"])}
        ev = events if events is not None else pd.DataFrame(columns=["time", "currency", "impact", "title"])
        if len(ev):
            ev = ev[ev["impact"].str.lower().isin(impacts)].copy()
            ev["time"] = pd.to_datetime(ev["time"], utc=True)
            ev = ev.sort_values("time").reset_index(drop=True)
        self.events = ev

    @property
    def has_data(self) -> bool:
        return self.enabled and len(self.events) > 0

    def covers(self, ts: pd.Timestamp) -> bool:
        """Is ts inside the date range the cached calendar knows about?"""
        if not self.has_data:
            return False
        return self.events["time"].iloc[0] - pd.Timedelta(days=1) <= ts <= \
            self.events["time"].iloc[-1] + pd.Timedelta(days=1)

    def _relevant(self, inst: Instrument) -> pd.DataFrame:
        cur = set(inst.currencies)
        if "XAU" in cur:
            cur.add("USD")
        return self.events[self.events["currency"].isin(cur | {"ALL"})]

    def blackout(self, inst: Instrument, ts: pd.Timestamp) -> tuple[bool, str]:
        """(blocked, description) for an entry at time ts."""
        if not self.has_data:
            return False, ""
        ev = self._relevant(inst)
        hit = ev[(ev["time"] - self.before <= ts) & (ts <= ev["time"] + self.after)]
        if len(hit):
            e = hit.iloc[0]
            return True, f"{e['currency']} {e['title']} at {e['time']:%a %H:%M} UTC"
        return False, ""

    def next_event_text(self, inst: Instrument, ts: pd.Timestamp, horizon_hours: int = 2) -> str:
        if not self.has_data:
            return "news calendar unavailable"
        if not self.covers(ts):
            return "no calendar coverage for this date"
        ev = self._relevant(inst)
        nxt = ev[(ev["time"] > ts) & (ev["time"] <= ts + pd.Timedelta(hours=horizon_hours))]
        if len(nxt):
            e = nxt.iloc[0]
            return f"{e['currency']} {e['title']} in {int((e['time'] - ts).total_seconds() // 60)} min"
        return f"no high-impact news within {horizon_hours}h"
