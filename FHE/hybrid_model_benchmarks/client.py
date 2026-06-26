"""Showcase for the hybrid model converter."""

import json
import os
import csv
import sys
import numpy as np
from time import perf_counter 
from pathlib import Path

import torch
from deploy import SimpleMLP
from concrete.ml.torch.hybrid_model import HybridFHEMode, HybridFHEModel


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if __name__ == "__main__":
    # Loads configuration dumped by deploy.py
    with open("hybrid_mdoel_benchmarks/configuration.json", "r") as file:
        configuration = json.load(file)

    module_names = configuration["module_names"]
    model_name = configuration["model_name"]
    print (module_names)
    device = "cpu"
    print(f"Using device: {device}")

    ####### change your model here if you want to
    parameters = int(sys.argv[1])
    layers = int(sys.argv[2])
    model = SimpleMLP(parameters,layers).eval()


    hybrid_model = HybridFHEModel(
        model,
        module_names,
        server_remote_address="http://0.0.0.0:8080",
        model_name=model_name,
        verbose=False,
    )

    path_to_clients = Path(__file__).parent / "clients"
    hybrid_model.init_client(path_to_clients=path_to_clients)
    hybrid_model.set_fhe_mode(HybridFHEMode.REMOTE)

    for module in hybrid_model.remote_modules.values():
        module.fhe_local_mode = HybridFHEMode.REMOTE    



    ################ In order to run inference on other stuff change code down below

    
    # Generate random input tensor with batch size 1 and 10 features
    data = []
    for i in range(1000+5):
        data.append(torch.randn((1,10), dtype=torch.float32))

    print("Running encrypted inference on random input...")

    times = []
    errors = []
    plain_model = SimpleMLP(parameters,layers).eval()


    counter = 0
    for i in data:

        start_time = perf_counter() 
        with torch.no_grad():
            prediction = model.forward(i)
        end_enc = perf_counter()


        begin = perf_counter()
        res_plain = plain_model.forward(i)
        end_plain = perf_counter()

        if counter > 5:
            times.append(end_enc-start_time)
            errors.append(np.mean((res_plain.detach().numpy() - prediction.numpy()) ** 2))
        counter += 1

    print(f"Inference took on average {np.average(times):.4f} seconds")
    print(f"the standard deviation is: {np.std(times):.4f}")

    file = open("results/mlp/hybrid/data.csv", "a", newline='')
    writer = csv.writer(file)
    writer.writerow(["number of parameters", "number of layers", "inference time enc [s]", "std enc [s]", "MSE"])
    writer.writerow([parameters, layers, round(np.mean(times),4), round(np.std(times),4), round(np.mean(errors),4)])
    file.close()