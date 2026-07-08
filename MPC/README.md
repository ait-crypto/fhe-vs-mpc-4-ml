# MPC Inference Benchmarks

A self-contained benchmark harness that measures **secure multi-party-computation
(MPC) inference vs. plaintext inference** for a range of neural-network
architectures. Secure execution runs on [SPU](https://github.com/secretflow/spu)
(ABY3, 3-party, `FM64` field); the plaintext baseline runs in ordinary JAX/Flax.

For each run the harness records wall-clock time, and — where supported —
CPU package/DRAM **power** (via RAPL), **memory** footprint, **loopback
bandwidth**, and the **RMSE** of the secure output against the plaintext output.

## Repository layout

```
.
├── benchmark_full_stats.py     # MLP / CNN / LSTM / LinearRegression sweeps + full resource stats
├── benchmark_classic_cnn.py    # LeNet / AlexNet / TinyCNN
├── benchmark_mobilefacenet.py  # MobileFaceNet
├── run_nodes.py                # brings up the local SPU node cluster
├── utils.py                    # small shared helpers (count_parameters)
├── configs/
│   └── 3pc.json                # 3-party ABY3 cluster + device topology
├── models/
│   ├── generic.py              # parametrised MLP / CNN / LSTM / LinearRegression
│   ├── tiny_cnn.py             # TinyCNN
│   └── mobilefacenet.py        # MobileFaceNet
├── pyproject.toml              # dependencies (uv)
├── requirements.txt            # dependencies (pip)
└── .python-version             # 3.10
```

## Requirements

- Python **3.10**
- The pinned packages in `pyproject.toml` / `requirements.txt`
  (`spu==0.9.3b0`, `flax==0.8.4`, `tqdm`, `psutil`, `pyrapl`). JAX is pulled in
  transitively by SPU/Flax.

> **Power measurement** relies on Intel/AMD RAPL through `pyRAPL`. It only works
> on Linux and typically needs read access to
> `/sys/class/powercap/intel-rapl/*` (often root, or a udev rule). If RAPL is
> unavailable the benchmarks still run — power columns are simply reported as
> `NaN`.

### Install

With [uv](https://docs.astral.sh/uv/) (matches how the results were produced):

```bash
uv venv --python 3.10
uv pip install -r requirements.txt
```

Or with plain pip:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

Secure benchmarks talk to a running SPU cluster. Open **two terminals**.

**Terminal 1 — start the nodes** (leave running):

```bash
python run_nodes.py -c configs/3pc.json up
```

**Terminal 2 — run a benchmark:**

```bash
# Full-stats MLP sweep, secure + plaintext, writing JSON results
python benchmark_full_stats.py --model mlp --use-spu --results mlp.json
```

Stop the cluster with `Ctrl-C` in terminal 1 when done.

## Benchmarks

### `benchmark_full_stats.py`

Sweeps a parametrised model family and records the full resource profile
(time, power, memory, bandwidth, RMSE). Sweep definitions live in the
`*_config` dicts near the bottom of the file — edit them to change sizes/shapes.

| flag | default | meaning |
| --- | --- | --- |
| `--model` | – | `mlp` \| `cnn` \| `lstm` \| `lin` |
| `--use-spu` | off | run the secure SPU path (omit for plaintext-only) |
| `--config` | `configs/3pc.json` | SPU cluster/device config |
| `--num-epochs-spu` | `1000` | number of secure inference runs |
| `--num-epochs-plain` | `10000` | number of plaintext inference runs |
| `--results` | – | output JSON path |

```bash
python benchmark_full_stats.py --model cnn --use-spu \
    --num-epochs-spu 500 --results cnn.json
```

### `benchmark_classic_cnn.py`

Times classic CNNs (secure vs. plaintext) and prints the means/stdevs.

```bash
python benchmark_classic_cnn.py --model lenet     # lenet | alexnet | tinycnn
```

### `benchmark_mobilefacenet.py`

Times MobileFaceNet (112×112×3 input) and writes timing JSON.

```bash
python benchmark_mobilefacenet.py --out-results mbf.json --num-epochs 10
```

## Configuration (`configs/3pc.json`)

Defines the node addresses and the device topology: an `SPU` device backed by
three nodes running **ABY3** over the **`FM64`** field, plus two `PYU`
(plaintext) parties `P1` (data owner) and `P2` (model owner). To change the
protocol/field, network latency, or ports, edit `runtime_config` /`nodes` and
pass the file via `--config`. All node addresses default to `127.0.0.1`
(localhost); point them at real hosts to benchmark over a network.

## Reproducing paper numbers

1. `python run_nodes.py -c configs/3pc.json up` (keep running).
2. Run each benchmark with the desired `--model` / sweep settings.
3. Collect the emitted JSON files. Each entry records `mean_s`/`mean_p`
   (secure/plaintext time), `rmse`, and — for `benchmark_full_stats.py` — the
   power/memory/bandwidth summaries and per-sample `timeseries`.

## Using this as an evaluation repo

This directory is standalone: copy it into your evaluation/artifact repository
as-is. It has no dependency on the surrounding project — all model definitions
and the node launcher are included here, and every script defaults to
`configs/3pc.json`, so it runs from its own root with no path edits.
