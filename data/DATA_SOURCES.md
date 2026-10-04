# Real-data sources (Phase 1, ingested 2026-10-04)

Two open tracking datasets, converted into the pipeline's trajectory format
(`data/trajectories/<TAG>_<id>.parquet`: period, frame, time, player_id, team,
x, y — x/y normalized 0–1, same schema as the Metrica `Sample_Game_*.parquet`).
Each parquet has a `<TAG>_<id>.meta.json` sidecar with fps, pitch, teams,
license, and counts.

## SkillCorner opendata — 20 matches (prefix `SK_`)
- Source: https://github.com/SkillCorner/opendata — 2024/25 Australian A-League,
  broadcast tracking at **10 fps** (+ dynamic events, phases of play, season
  aggregates incl. SkillCorner's own off-ball-run annotations).
- Raw: `data/raw/skillcorner/<id>_{match.json,tracking_extrapolated.jsonl,dynamic_events.csv,phases_of_play.csv}`
- Native coords: meters, pitch center origin. Normalized x/L+0.5, y/W+0.5.
- Ball present in ~73% of frames (broadcast tracking loses it); player points
  are ~52–66% detected, rest extrapolated (see `detected_fraction` in sidecars).
- License: repo LICENSE is **MIT** (2020 SkillCorner); README asks users to
  credit SkillCorner. No separate data-license file found. (A third-party
  listicle claimed CC-BY-NC — NOT confirmed; repo itself says MIT.)
- Converter: `scripts/convert_skillcorner.py --match <id>`

## DFL / IDSSE (Bassek et al. 2025) — 7 matches (prefix `DFL_`)
- Source: HuggingFace `pysport/idsse-data` (figshare DOI
  10.6084/m9.figshare.28196177; paper Sci Data 12, 195 (2025)).
  Bundesliga 2022/23: Köln–Bayern, Bochum–Leverkusen, + 5× Düsseldorf home games.
- **25 Hz full optical tracking** — ball present in 100% of frames.
- Raw: `data/raw/dfl/DFL_{02_01_matchinformation,03_02_events_raw,04_03_positions_raw_observed}_<comp>_DFL-MAT-<id>.xml`
- Native coords: meters, pitch center origin. Events XML kept raw (corner-origin
  0–105/0–68) for future event fusion.
- License: **CC-BY 4.0 — commercial-safe** (confirmed in kloppy docstring + paper).
- Converter: `scripts/convert_dfl.py --match <id>`

## Totals
27 matches · 43.1M trajectory rows · 1.88M frames · ~5.3 GB on disk.

## Notes for retraining
- Downstream scripts hardcode `FPS = 25.0` (`scripts/build_windows.py`); the
  75/25-frame windows assume 25 fps. DFL matches that natively; SkillCorner's
  10 fps needs fps-aware windowing (or resampling) before reuse.
- `attack_dirs()` derives attack direction from ball displacement per
  (period, team), so both datasets' native orientation is fine as-is.
- Suggested match-level split: train on DFL (7) + 13 SK, validate on 4 SK,
  test on 3 SK — or leave-one-competition-out.
- kloppy's remote loaders (`load_open_data`) do NOT work in this sandbox:
  fsspec ignores the egress proxy env vars. Download with curl, then use
  kloppy's local-file loaders (or the converters above, which parse natively).
