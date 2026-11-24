# AgentQuote: Multi-Dimensional Cost Quoting for Pay-as-You-Go Data Agents

AgentQuote prices a data-agent session before it runs. For one request on a pay-as-you-go platform such as Earth Engine or BigQuery, it returns a quote: an upper bound on platform compute, LLM tokens and wall time, and all three bounds hold together with a probability you choose. It then picks the cheapest agent configuration whose quote fits your budget, with a certified cap on the share of requests that end up wrong or over their quote. If no configuration qualifies, it declines.

An agent does not run its plan once. It looks up the catalogue, inspects data, fails, repairs and retries, and the platform bills every one of those calls. A price for a single clean run misses most of that spend. AgentQuote prices the clean run with a domain adapter and imagines the rest of the session with a world model learned from logged sessions.

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

A session's conformal score is its largest ratio of realised cost to estimate across meters. One quantile of that score scales every estimate, so the three bounds hold jointly, not one at a time. The choice threshold is tested on held-out sessions with fixed-sequence binomial tests, so the risk cap holds with high probability.

## Installation

Use **Python 3.12** for the environment below. The code requires **Python 3.10 or newer**.

We encourage you to use conda or miniconda:

```sh
conda create -n agentquote python=3.12 pip -y
conda activate agentquote
```

Clone the repository and install the dependencies from its root:

```sh
git clone git@github.com:phkhanhtrinh23/AgentQuote.git
cd AgentQuote
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) lists NumPy, SciPy, pandas, scikit-learn, LightGBM, Matplotlib and sqlglot, plus the OpenAI client and the Google Cloud clients for live collection.

### API keys

Create a local `.env` file using [.env.example](.env.example) as a reference:

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

Replace the placeholders with your own key and project. `OPENROUTER_BASE_URL` can point to any compatible provider, and `LOCAL_BACKBONES` sends the listed backbones to a local server such as vLLM. Keys are only needed to collect live sessions.

## Data Sources

See [data/README.md](data/README.md) for how to build request sets and collect sessions from the sources below.

| Source | What it provides |
|---|---|
| [Spider 2.0-Lite](https://github.com/xlang-ai/Spider2) | BigQuery text-to-SQL questions over public databases, with gold results |
| Cloud-Based Geospatial Benchmark (CBGB) | Earth Engine questions with numeric gold answers |
| UnivEARTH | Earth Engine yes/no questions |
| [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) | Hand-labelled flood chips with event dates |
| [Natural Earth](https://www.naturalearthdata.com/) | Country shapes for drawing analysis regions |

The built-in testbeds generate metered Earth Engine and BigQuery sessions in the same format, so no download is needed to run the pipeline.

### Input format for logged sessions

- **Requests:** `data/<platform>/requests.jsonl`, one request per line with its unit (region or database), type, text, candidate configurations and optional reference.
- **Sessions:** `data/<platform>/sessions.jsonl`, one session per line for one configuration on one request: the draft, exploration steps, attempts with outcomes and metered calls, inspections, and the final answer with its error.
- **Meters:** every turn records input and output tokens, observation size and latency. Every platform call records its metered cost.

## Running

Run the whole pipeline from the repository root:

```sh
bash run_all.sh
```

It runs the tests, generates session pools on both testbeds, fits and calibrates AgentQuote and the baselines over random unit splits, runs every analysis and draws the figures. A full run takes 1 to 2 hours. `QUICK=1 bash run_all.sh` runs the same pipeline on smaller pools in about 5 minutes.

The run writes to `results/`:

- `<platform>/quotes.json`: joint and per-meter coverage, width and miss of every quoting method, and paired width reductions
- `<platform>/choice.json`: risk, cost, admissibility and declines of every choice rule
- `<platform>/ablation.json`, `shift.json`, `sensitivity.json`, `breakdown.json`, `efficiency.json`, `case.json`
- `summary.md`: the main tables
- `figures/`: PDF and PNG figures

To run one analysis on sessions already in `data/`:

```sh
python -m agentquote.experiments prepare --data data --out results
python -m agentquote.experiments quotes  --data data --out results
```

Tasks are `breakdown`, `quotes`, `choice`, `ablation`, `shift`, `efficiency`, `sensitivity`, `case`, `figures` and `all`. `prepare` fits and caches every split once, so later tasks reuse it.

To run on your own logged sessions instead of the testbeds:

```sh
SKIP_GENERATE=1 bash run_all.sh
```

Other flags:

- `--assign`: number of random unit splits (default 10).
- `--n-roll`: imagined rollouts per configuration (default 1,000).
- `--platform`: `earthengine`, `bigquery` or `all`.
- `--seed`: base random seed.

`run_all.sh` reads the same settings from `ASSIGN`, `NROLL`, `SCALE` (testbed pool size), `DATA`, `OUT` and `PY`.

[`Settings`](agentquote/config.py) holds the error levels (α = 0.1, δ = 0.2, η = 0.1), the error tolerance τ = 0.1, the attempt and turn caps, and the model sizes. Platform and token prices are in the same file.

To use AgentQuote from Python:

```python
from agentquote.adapters import get_adapter
from agentquote.pipeline import AgentQuote

aq = AgentQuote(get_adapter("earthengine")).fit(fit_sessions)
cal = aq.calibrate(aq.estimate(quote_sessions, rng), aq.estimate(choice_sessions, rng))
quotes, choice = aq.quote_and_choose(aq.estimate(candidates, rng), cal, budget)
```

To add a platform, subclass [`DomainAdapter`](agentquote/adapters/base.py) with a compiler, operator classes, prices and outcome classes. The world model, quote and choice stay the same.

Live collection stops once spend passes `--max-usd`. The session in progress finishes first, so the total can go over the cap by up to one session. Each session's exact spend is logged.

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
├── data/                  guide to preparing session data
├── run_all.sh             end-to-end run
├── requirements.txt       Python dependencies
├── .env.example           template for API keys and cloud project
└── LICENSE                MIT license
```

## License

MIT. See [LICENSE](LICENSE). External datasets, platforms and model APIs keep their own terms.
