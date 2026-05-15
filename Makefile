.PHONY: all fetch dataset train charts test clean

all: fetch dataset train charts

fetch:
	python3 scripts/fetch.py --start 2018 --end 2023 --out data

dataset:
	python3 scripts/build_dataset.py --data data --out data/plays.parquet

train:
	python3 scripts/train.py --data data/plays.parquet --out-dir models

charts:
	python3 scripts/charts.py

test:
	python3 -m pytest tests/ -v

clean:
	rm -f data/*.csv data/*.parquet models/* charts/*.png
