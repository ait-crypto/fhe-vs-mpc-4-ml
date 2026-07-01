# FHE vs. MPC for ML

This repository is the collection of code used in the paper .... . It utilizes uv managed python virtual environments and implements all FHE and MPC benchmark codes, as well as artifacts used in our paper.


## Installation

To use the the repository clone it into a desired folder and then access it. In the command line, go into the respective folder you want to use and sync the virtual environment, which will download all dependencies, with the following command:
```
uv sync
``` 

After that, all benchmarks can be executed with the following commands:

Using uv: 
```
uv run python3 path/to/filename
``` 
or

activate the python virtual environment: 
```
source .venv/bin/activate
``` 
and then run 
```
python3 path/to/filename
```

All scripts are meant to be executed with the working directory being either FHE, MPC or Plots.
Some files may require args to be appended to the execution command. A list of all of those can be found below with the intended usage.

| file | parameters | example |
| ---- | ---------- | ------- |
| cosine.py | vector size | concrete_benchmarks/cosine.py 16|
| l1.py | vector size | concrete_benchmarks/l1.py 16 |
| l2.py | vector size | concrete_benchmarks/l2.py 16 |
| deploy.py | model parameter amount, layer amount | hybrid_model_benchmarks/deploy.py 1000 3|
| client.py | model parameter amount, layer amount | hybrid_model_benchmarks/client.py 1000 3|