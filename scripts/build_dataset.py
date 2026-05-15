"""Roll up the raw nflfastR CSVs into a clean modeling table.

For each offensive snap we keep the game state (down, distance, field position,
score margin, clock, timeouts) and the eventual game outcome from the offense's
point of view. Ties and not-yet-decided games are dropped.
"""

import argparse
import glob
import os

import duckdb

FEATURES = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "qtr",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]


def build(data_dir, out_path):
    files = sorted(glob.glob(os.path.join(data_dir, "pbp_*.csv")))
    if not files:
        raise SystemExit(f"no pbp_*.csv files in {data_dir}")
    print(f"reading {len(files)} season files")

    con = duckdb.connect()
    file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    con.execute(
        f"create view raw as select * from read_csv_auto({file_list}, sample_size=-1, union_by_name=true)"
    )

    feature_cols = ", ".join(FEATURES)
    q = f"""
    select
        game_id, season, week, posteam, defteam, home_team, away_team,
        {feature_cols},
        case
            when posteam = home_team and result > 0 then 1
            when posteam = away_team and result < 0 then 1
            else 0
        end as posteam_won
    from raw
    where play_type in ('pass', 'run', 'punt', 'field_goal')
      and posteam is not null
      and down is not null
      and yardline_100 is not null
      and score_differential is not null
      and game_seconds_remaining is not null
      and result is not null
      and result <> 0
    """
    df = con.execute(q).df()
    print(f"  rows: {len(df):,}")
    print(f"  games: {df['game_id'].nunique():,}")
    print(f"  seasons: {sorted(df['season'].unique().tolist())}")
    print(f"  win rate from posteam POV: {df['posteam_won'].mean():.3f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_parquet(out_path, index=False)
    sz = os.path.getsize(out_path) / 1e6
    print(f"  wrote {out_path} ({sz:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/plays.parquet")
    args = ap.parse_args()
    build(args.data, args.out)


if __name__ == "__main__":
    main()
