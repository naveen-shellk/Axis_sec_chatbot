"""
agent/core/dates.py
--------------------
Shared date parsing / normalisation helpers.

These were previously duplicated byte-for-byte in several flow handlers
(statement, brokerage). Centralising them removes drift risk — one place to fix
a date-format bug — with no behaviour change (same regexes, same return types,
all DD-MM-YYYY / DD/MM/YYYY day-month-year order).
"""

from __future__ import annotations

import re
from datetime import date, timedelta


def parse_date(text: str) -> date | None:
    """Parse the FIRST DD-MM-YYYY or DD/MM/YYYY date found in `text`."""
    m = re.search(r"\b(\d{2})[-/](\d{2})[-/](\d{4})\b", text or "")
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    return None


def parse_range(text: str) -> tuple[date, date] | None:
    """Extract TWO DD-MM-YYYY/DD-MM-YYYY dates from free text as (start, end),
    ordered so start <= end. Returns None if fewer than two valid dates."""
    dates = re.findall(r"\b(\d{2})[-/](\d{2})[-/](\d{4})\b", text or "")
    if len(dates) >= 2:
        try:
            d1 = date(int(dates[0][2]), int(dates[0][1]), int(dates[0][0]))
            d2 = date(int(dates[1][2]), int(dates[1][1]), int(dates[1][0]))
            return (d1, d2) if d1 <= d2 else (d2, d1)
        except ValueError:
            return None
    return None


def norm_ddmmyyyy(s: str) -> str | None:
    """Normalise a loose date string (D-M-YYYY / DD/MM/YYYY) to DD-MM-YYYY,
    or None if unparseable."""
    s = (s or "").strip()
    m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", s)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1))).strftime("%d-%m-%Y")
        except ValueError:
            return None
    return None


def preset_range(label: str) -> tuple[date, date] | None:
    """Map a preset label ('Last 7/15/30 days') to a (start, end) range ending
    today. Returns None for anything else."""
    days = {"last 7 days": 7, "last 15 days": 15, "last 30 days": 30}.get(
        (label or "").strip().lower())
    if not days:
        return None
    today = date.today()
    return (today - timedelta(days=days - 1), today)
