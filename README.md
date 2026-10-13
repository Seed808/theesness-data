# theesness-data

Raw data collected on a schedule by GitHub Actions for the TheEsness Predictions model.

| Workflow | When | Saves |
|---|---|---|
| collect-lines-and-injuries | every 30 min | ESPN scoreboards + betting lines for NFL, NCAAF, NBA, CBB, MLB, NHL (today + 7 days), NBA/NFL injury lists, newest official NBA injury report PDF |
| mlb-statcast-and-schedule | daily 08:13 UTC | Statcast pitches (season to date, by month), MLB probable pitchers and lineups |
| nba-injury-backfill | by hand, once | Official NBA injury report PDFs for a past date range |

Files are raw (JSON / PDF / parquet). The model parses them. `first.json` in each folder is the first
snapshot ever seen for that day: the line projections are locked against.
