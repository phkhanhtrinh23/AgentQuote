# AgentQuote: Multi-Dimensional Cost Quoting for Pay-as-You-Go Data Agents

AgentQuote prices a data-agent session before it runs. For one request on a pay-as-you-go platform such as Earth Engine or BigQuery, it returns a quote: an upper bound on platform compute, LLM tokens and wall time, and all three bounds hold together with a probability you choose. It then picks the cheapest agent configuration whose quote fits your budget, with a certified cap on the share of requests that end up wrong or over their quote. If no configuration qualifies, it declines.

An agent does not run its plan once. It looks up the catalogue, inspects data, fails, repairs and retries, and the platform bills every one of those calls. A price for a single clean run misses most of that spend. AgentQuote prices the clean run with a domain adapter and then imagines the rest of the session with a world model learned from logged sessions.

## Architecture

```text
Request, candidate configurations (backbone, data option, method), budget
        │
        ▼
Cheap drafter writes a plan for each configuration
        │
        ▼
Domain adapter
        ├── Compiles the draft into typed operators
        └── Prices one clean run on every meter
        │
        ▼
Session world model (fitted on logged sessions)
        ├── Policy: explore, inspect, give up
        ├── Transitions: failure hazard with frailty, failure outcome, plan edits
        └── Costs: per-operator compute, re-read tokens, latency
        │
        ▼
Rollouts → per-meter estimate of the whole session
        │
        ▼
Joint conformal quote (one factor scales all meters)
        │
        ▼
Certified choice (error-risk score + Learn-then-Test threshold)
        │
        ▼
Report: quote per configuration, chosen configuration or decline
```

The conformal score of a session is its largest ratio of realised cost to estimate across meters. One quantile of that score scales every estimate, so the three bounds hold jointly, not one at a time. The choice threshold is tested on a held-out set with fixed-sequence binomial tests, so the risk cap holds with high probability over that set.

## Installation

Use **Python 3.12** for the environment below. The code requires **Python 3.10 or newer**.

We encourage you to use conda or miniconda:

```sh
conda create -n agentquote python=3.12 pip -y
conda activate agentquote
```

Clone the repository and install it from its root:

```sh
git clone git@github.com:phkhanhtrinh23/AgentQuote.git
cd AgentQuote
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

[requirements.txt](requirements.txt) lists NumPy, SciPy, pandas, scikit-learn, LightGBM, Matplotlib and sqlglot. To collect sessions on the live platforms, also install the extras:

```sh
python -m pip install -e ".[live]"
```

### API keys

Live collection needs an LLM endpoint and a Google Cloud project. Create a local `.env` file using [.env.example](.env.example) as a reference:

```sh
cp .env.example .env
```

```dotenv
OPENROUTER_API_KEY=your-api-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LOCAL_BACKBONES=
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
GOOGLE_CLOUD_PROJECT=your-project
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

Any OpenAI-compatible endpoint works. `LOCAL_BACKBONES` routes the listed backbones to a local server such as vLLM.

## Data Sources

| Source | What it provides |
|---|---|
| [Spider 2.0-Lite](https://github.com/xlang-ai/Spider2) | BigQuery text-to-SQL questions over public databases, with gold results |
| Cloud-Based Geospatial Benchmark (CBGB) | Earth Engine questions with numeric gold answers |
| UnivEARTH | Earth Engine yes/no questions |
| [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) | Hand-labelled flood chips with event dates |
| [Natural Earth](https://www.naturalearthdata.com/) | Country shapes for drawing analysis regions |

Place the files under `data_cache/` (for example `data_cache/spider2/spider2-lite`, `data_cache/cbgb/eval_set.csv`, `data_cache/univearth/data.json`). The loaders in [agentquote/live/](agentquote/live/) list the exact paths.

No downloads are needed to try the method. The built-in testbeds generate metered Earth Engine and BigQuery sessions in the same log format.

### Session log format

Each line of `sessions.jsonl` is one session of one configuration on one request. It holds the drafter's turn and compiled plan, the exploration steps, every attempt with its outcome, executed plan and metered calls, the inspections between attempts, the final answer and its error against the reference. Each turn records input and output tokens, observation size and latency. Each platform call records its metered cost. See [agentquote/types.py](agentquote/types.py).

## Running

Run the whole pipeline:

```sh
bash run_all.sh             # full run, 1-2 hours
QUICK=1 bash run_all.sh     # smaller pools, 2 random splits, 200 rollouts, about 5 minutes
```

This generates session pools on both testbeds, fits AgentQuote and the baselines over random splits, runs every analysis and draws the figures. Results go to `results/`: one JSON file per analysis and platform, `summary.md` with the main tables, and `figures/`.

Run single steps:

```sh
python scripts/generate_testbed.py --out data
python -m agentquote.experiments prepare --data data --out results
python -m agentquote.experiments quotes  --data data --out results
```

Tasks: `breakdown`, `quotes`, `choice`, `ablation`, `shift`, `efficiency`, `sensitivity`, `case`, `figures`. Each takes `--assign` (number of random splits) and `--n-roll` (rollouts per configuration).

### Collecting real sessions

```sh
python scripts/build_ee_requests.py --out data/earthengine
python scripts/collect_sessions.py --platform earthengine --out data/earthengine --max-usd 50
python scripts/collect_sessions.py --platform bigquery    --out data/bigquery    --max-usd 50
SKIP_GENERATE=1 bash run_all.sh
```

Collection spends real money. It stops at `--max-usd` and resumes where it left off.

### Using it in code

```python
from agentquote.adapters import get_adapter
from agentquote.pipeline import AgentQuote

aq = AgentQuote(get_adapter("earthengine")).fit(fit_sessions)
cal = aq.calibrate(aq.estimate(quote_sessions, rng), aq.estimate(choice_sessions, rng))
quotes, choice = aq.quote_and_choose(aq.estimate(candidates, rng), cal, budget)
```

To add a platform, subclass [`DomainAdapter`](agentquote/adapters/base.py): a compiler, operator classes, prices and outcome classes. The world model, quote and choice stay the same.

[`Settings`](agentquote/config.py) holds the defaults: α = 0.1, δ = 0.2, η = 0.1, τ = 0.1, 1,000 rollouts, at most 5 attempts and 50 turns. Platform and token prices are in the same file.

## Project Structure

```text
AgentQuote/
├── agentquote/
│   ├── adapters/          Earth Engine and BigQuery: compile, price, classify outcomes
│   ├── worldmodel.py      session world model: policy, transition and cost heads, rollouts
│   ├── conformal.py       joint conformal quote and its variants
│   ├── choice.py          error-risk score, choice rule, Learn-then-Test
│   ├── baselines.py       clean-run, per-group, self-estimate and learned-quantile baselines
│   ├── pipeline.py        AgentQuote end to end
│   ├── testbed/           metered Earth Engine and BigQuery session environments
│   ├── live/              agent loop, LLM client and platform executors
│   ├── experiments/       analyses and figures
│   ├── config.py          settings, backbones and prices
│   └── types.py           requests, plans and session logs
├── scripts/               generate, collect and summarize
├── tests/                 unit tests
├── run_all.sh             end-to-end run
├── requirements.txt       Python dependencies
├── .env.example           template for API keys and cloud project
└── LICENSE                MIT license
```

## License

MIT. See [LICENSE](LICENSE). External datasets, platforms and model APIs keep their own terms.
