
# FHE Inference Benchmarks

This directory contains all FHE benchmarks utilizing the concrete-ml and concrete open source FHE libraries. Plaintext baselines run either with SKlearn or torch models.
Each run records inference times, plain and encrypted, their standard deviation and error metrics. Furthermore most of the benchmarks implement power consumption trackers using the powerstat command via RAPL interface.

## Repository layout
```
.
├── basic_layer_benchmarks/         # benchmarks for basic layers of neural machine learning models
|   ├── gelu.py
|   ├── relu.py
|   ├── sigmoid.py
|   └── matmul.py
├── concrete_benchmarks/            # benchmarks of different distance measures using the concrete library
|   ├── cosine.py
|   ├── l1.py
|   └── l2.py
├── hybrid_model_benchmarks/        # hybrid model benchmark implementations
|   ├── client.py               
|   ├── deploy.py
|   └── server.py  
├── model_benchmarks/               # benchmarks of different models (cnn, mlp, random forest, linear regression & logistic classification)
|   ├── cnn.py
|   ├── linear_regression.py
|   ├── linear_trained.py           # trained version of the linear regression with accuracy metrics
|   ├── logistic_classification.py
|   ├── mlp.py
|   ├── mlp_trained.py              # trained version of the mlp with accuracy metrics
|   └── random_forest_regressor.py
├── results/                        # directory where all results are saved
|   └── ...                         # sub directories for the respective benchmarks
├── pyproject.toml                  # dependencies (uv)
├── requirements.txt                # dependencies (pip)
└── .python-version                 # 3.10
``` 

## Requirements

- **Python 3.10**
- all libraries listed under dependencies in the `pyproject.toml`/`requirements.txt`
> **Power Masurements:** To benchmark power consumption the powerstat command needs to be installed as a prerequisit. This can be done via the respective package manager in Linux. For users on Windows and MAC, the benchmarks needs to be adapted to exclude this metric, as it isn't available on these operating systems. 

### Install

With [uv](https://docs.astral.sh/uv/) (matches how the results were produced):

```bash
uv sync
```

Or with plain pip:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
``` 

## Benchmarking

Run benchmarks from the FHE folder either using uv:

```bash
uv run python3 path/to/filename
``` 
or activate the python virtual environment if not yet done:

``` bash
source .venv/bin/activate
``` 
and then use the python3 command 

``` bash
python3 path/to/filename
```

Some files may require args to be appended to the execution command. A list of all of those can be found below with the intended usage.

| file | parameters | example |
| ---- | ---------- | ------- |
| cosine.py | vector size | concrete_benchmarks/cosine.py 16|
| l1.py | vector size | concrete_benchmarks/l1.py 16 |
| l2.py | vector size | concrete_benchmarks/l2.py 16 |
| deploy.py | model parameter amount, layer amount | hybrid_model_benchmarks/deploy.py 1000 3|
| client.py | model parameter amount, layer amount | hybrid_model_benchmarks/client.py 1000 3|
| linear_regression.py | quantization bits | model_benchmarks/linear_regression.py 7|
| linear_trained.py | quantization bits | model_benchmarks/linear_trained.py 3|
| logistic_classification.py | quantization bits | model_benchmarks/logistic_classification.py 11|
| mlp_trained.py | quantization bits | model_benchmarks/mlp_trained.py 6|


## Hybrid model benchmarks.

The hybrid model benchmarks require a small setup. In order to be able to run the benchmarks the model has first to be created and serialized. This is done by running `deploy.py` with the desired mlp paramater and layer amount as args. This will create some subfolders storing configuration details, the compiled layers and such. After that start one terminal and run `server.py` first. Then `client.py` can be executed with the same mlp model parameter and layer configuration as used to deploy the model. The client will create additional folders for storing keys and execution layers. Before benchmarking a new model configuration all folders and fiels created by `deploy.py` need to be deleted. It is further recommended to delete folders created by `client.py` from time to time, as they can get pretty large very quickly.    
