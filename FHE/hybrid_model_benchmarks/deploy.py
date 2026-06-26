"""Showcase for the hybrid model converter."""
import torch
import json
import os
import torch
import math
import sys
import re
from torch import nn
from pathlib import Path
import brevitas.nn as qnn
from concrete.ml.torch.hybrid_model import HybridFHEModel
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat



N_BITS = 6

class SimpleMLP(nn.Module):
    def __init__(self, params, layers:int=1, qlinear_args={
            "weight_bit_width": N_BITS,
            "weight_quant": Int8WeightPerTensorFloat,
            "bias": False,
            "narrow_range": True
        }, qidentity_args={"bit_width": N_BITS, "act_quant": Int8ActPerTensorFloat}):

        lin_param = 0
        if (layers > 1):
            lin_param = int(abs(round((-10+math.sqrt(10**2-(4*(layers-1)*-params)))/(2*(layers-1)))))
        else:
            lin_param = int(params/10)


        super(SimpleMLP, self).__init__()

        layer_list = []
        for i in range(layers):
            if (i == 0):
                layer_list.append(nn.Linear(10, lin_param))
            else:
                layer_list.append(nn.Linear(lin_param, lin_param))
            layer_list.append(nn.ReLU())

        self.all_layers = nn.Sequential(*layer_list)


    def forward(self, x):
        x = self.all_layers(x)
        return x





def compile_model(
    model_name: str,
    model,
    inputs: torch.Tensor,
    module_names: list,
    models_dir: Path):

    hybrid_model = HybridFHEModel(model, module_names, model_name=model_name, verbose=1)
    # Compile hybrid model
    hybrid_model.compile_model(
        inputs,
        n_bits=7,
    )

    # Save model for serving
    models_dir.mkdir(exist_ok=True)
    model_dir = models_dir / model_name
    print(f"Saving to {model_dir}")
    via_mlir = bool(int(os.environ.get("VIA_MLIR", 1)))
    hybrid_model.save_and_clear_private_info(model_dir, via_mlir=via_mlir)


if __name__ == "__main__":

    parameters = int(sys.argv[1])
    layers = int(sys.argv[2])

    ## change your model here
    net = SimpleMLP(parameters, layers).eval()
    model_name = "SimpleMLP"
    x = torch.randn((1, 10), dtype=torch.float32)


    # to extract different sub modules change the if condition in the loop below
    encrypted_modules = []
    for (k,_) in net.named_modules():
        print(k)
        #if k.find("relu") == -1:
        if re.search("[13579]$", k) == None:
            encrypted_modules.append(k)
        
    print(encrypted_modules)
    encrypted_modules = encrypted_modules[2:]
    models_dir = Path(__file__).parent / "compiled_models"
    models_dir.mkdir(exist_ok=True)

    compile_model(
        model_name,
        net,
        x,
        encrypted_modules,
        models_dir=models_dir,
    )

    configuration = {
        "model_name": model_name,
        "module_names": encrypted_modules,
    }

    with open("hybrid_model_benchmarks_configuration.json", "w") as file:
        json.dump(configuration, file)