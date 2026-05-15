"""Download nflfastR play-by-play CSVs for a range of seasons."""

import argparse
import os
import urllib.request

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv"


def fetch(years, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for y in years:
        path = os.path.join(out_dir, f"pbp_{y}.csv")
        if os.path.exists(path):
            print(f"  have {y}")
            continue
        url = URL.format(year=y)
        print(f"  fetching {y} from {url}")
        urllib.request.urlretrieve(url, path)
        sz = os.path.getsize(path) / 1e6
        print(f"    wrote {path} ({sz:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    fetch(range(args.start, args.end + 1), args.out)


if __name__ == "__main__":
    main()
