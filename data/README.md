# Data

The pipeline reads one folder per platform:

```text
data/
├── earthengine/
│   ├── requests.jsonl
│   └── sessions.jsonl
└── bigquery/
    ├── requests.jsonl
    └── sessions.jsonl
```

There are two ways to fill it.

## Testbed sessions

```sh
python scripts/generate_testbed.py --out data
```

This writes both platforms. It also writes an independently drawn second pool (`data/pool2/`), a pool priced by a weaker adapter (`data/hard/`), and the Earth Engine re-executions and own-draft sessions used by the shift analysis. `--scale 0.3` gives a smaller pool.

## Live sessions

Download the sources into `data_cache/` first:

| Source | Path |
|---|---|
| [Spider 2.0-Lite](https://github.com/xlang-ai/Spider2) | `data_cache/spider2/spider2-lite/` (the `spider2-lite` folder of the repository) |
| Cloud-Based Geospatial Benchmark (CBGB) | `data_cache/cbgb/eval_set.csv` |
| UnivEARTH | `data_cache/univearth/data.json` |
| [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) | `data_cache/sen1floods11/LabelHand/` and `data_cache/sen1floods11/meta.geojson` |
| [Natural Earth](https://www.naturalearthdata.com/) | `data_cache/naturalearth/countries_110m.geojson` |

Build the Earth Engine requests. These are the CBGB and UnivEARTH questions plus templated analyses (buildings, built-up fraction, tree cover, flood extent, NDVI change) over regions drawn from the land pool, each with several data options and methods. With Earth Engine access, the script also computes the layer references:

```sh
python scripts/build_ee_requests.py --out data/earthengine
```

BigQuery requests come straight from Spider 2.0-Lite: 205 questions over 70 public databases.

Run every candidate configuration once and log the session:

```sh
python scripts/collect_sessions.py --platform earthengine --out data/earthengine --max-usd 50
python scripts/collect_sessions.py --platform bigquery    --out data/bigquery    --max-usd 50
```

Each session runs under its own workload tag (Earth Engine) or job label (BigQuery), so its platform cost is read from billing, not guessed. Sessions already in `sessions.jsonl` are skipped, so a stopped run resumes.

## Formats

- **Requests:** one JSON object per line with `request_id`, `unit` (region or database), `rtype` (request type), `text`, `platform`, `candidates` (configuration keys `backbone|data|method`), and an optional `reference` with its `ref_kind` (`gold` or `layer`).
- **Sessions:** one JSON object per line for one configuration on one request. It holds the drafter's turn and compiled plan, the exploration steps, the attempts with their outcomes, executed plans and metered calls, the inspections between attempts, the final answer and its `error` against the reference.

Every turn records input and output tokens, the size of what the agent saw, and latency. Every platform call records its metered cost. [agentquote/types.py](../agentquote/types.py) has the full schema.
