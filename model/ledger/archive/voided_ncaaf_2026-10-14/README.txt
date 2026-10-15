Voided 2026-10-14 ~11:00 HST, before any of these games kicked off.
Reason: the ESPN line parser for college football negated the spread twice ("UGA -20.5" was stored as Georgia +20.5),
so every pick and edge in this file was computed against the wrong side of the line. Fixed in espn_ncaaf_slate;
Week 7 re-locked against the same first-seen ESPN lines with the correct sign.
