
from time import perf_counter, sleep
import torch
import csv
import subprocess
import math
import numpy as np
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from concrete.ml.torch.compile import compile_brevitas_qat_model
from torch import nn


N_BITS = 6

plain_times = []
plain_times_std = []
mean_error = []
cipher_times = []
cipher_times_std = []
sample_sizes = []
parameter_amounts = []
layer_amounts = []

## model used for quantize ciphertext inference
class SimpleMLP(nn.Module):
    def __init__(self, params, layers:int=1, qlinear_args={
            "weight_bit_width": N_BITS,
            "weight_quant": Int8WeightPerTensorFloat,
            "bias": False,
            #"bias_quant": False,
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
                layer_list.append(qnn.QuantIdentity(**qidentity_args))
                layer_list.append(qnn.QuantLinear(10, lin_param, **qlinear_args))
            else:
                layer_list.append(qnn.QuantLinear(lin_param, lin_param, **qlinear_args))
            layer_list.append(qnn.QuantReLU())

        self.all_layers = nn.Sequential(*layer_list)


    def forward(self, x):
        x = self.all_layers(x)
        return x


## model used for plaintext in 
class OSimpleMLP(nn.Module):
    def __init__(self, params, layers:int=1):

        lin_param = 0
        if (layers > 1):
            lin_param = int(abs(round((-10+math.sqrt(10**2-(4*(layers-1)*-params)))/(2*(layers-1)))))
        else:
            lin_param = int(params/10)

        super(OSimpleMLP, self).__init__()

        layer_list = []
        for i in range(layers):
            if i == 0: 
                layer_list.append(nn.Linear(10,lin_param))
            else: 
                layer_list.append(nn.Linear(lin_param, lin_param))
            layer_list.append(nn.ReLU())
        self.all_layers = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.all_layers(x)
        return x


def make_data(sample_size):
    data = []
    for i in range(sample_size):
        data.append(torch.randn((1,10)))
    
    return data


def test(layers, params, sample_size, mem_file):
    model = SimpleMLP(params, layers).eval()
    fhe_model = compile_brevitas_qat_model(model, torch.randn((1,10)), rounding_threshold_bits=N_BITS, n_bits=N_BITS)
    no_quan_model = OSimpleMLP(params, layers).eval()
    print("Compilation done")

    proc = bench_power_measurement(mem_file)   
    model_inference(no_quan_model, fhe_model, sample_size)
    proc.terminate()
    proc.wait()

    parameter_amounts.append(params)
    layer_amounts.append(layers)


def model_inference(model, fhe_model, sample_size):
    test_data = make_data(sample_size)

    ptimes= []
    ctimes = []
    errors = []

    x = 0
    for i in test_data:
        begin = perf_counter()
        res_plain = model.forward(i)
        end_plain = perf_counter()-begin

        begin = perf_counter()
        res = fhe_model.forward(i.numpy(), fhe="execute")
        end_enc = perf_counter()-begin

        if x > 1:
            res_plain = res_plain.detach().numpy()[0]
            res = res[0]
            err = np.sum((res - res_plain) ** 2) / len(res)
            errors.append(err)
            ptimes.append(end_plain)
            ctimes.append(end_enc)
            for j in range(100):
                begin = perf_counter()
                res_plain = model.forward(i)
                ptimes.append(perf_counter()-begin)
        x += 1

    print(np.mean(errors))
    avgC = np.mean(ctimes)
    std_devC = np.std(ctimes)
    avgP = np.mean(ptimes)
    std_devP = np.std(ptimes)

    print(std_devC, ", ", avgC)
    print(std_devP, ", ", avgP)

    if (std_devC*10 > avgC): return model_inference(model, fhe_model, int(sample_size*1.5))
    else:
        plain_times.append(avgP)
        cipher_times.append(avgC)
        plain_times_std.append(std_devP)
        cipher_times_std.append(std_devC)
        mean_error.append(np.mean(errors))
        sample_sizes.append(sample_size)
    
#################################################################################
#####################simple bench

def plain_bench(sample_size, params, layers):
    test_data = make_data(sample_size)
    model = OSimpleMLP(params, layers).eval()

    for i in test_data:
        model.forward(i)

#################################################################################
##################### power stuff
def base_power_measurement():

    cmd = ["sudo", "powerstat", "-R", "1", "60"]

    print("Running base powerstat...")
    with open(f"results/mlp/basic/idle_powerstat.txt", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print("Done. Output saved.")

def bench_power_measurement(f):

    cmd = ["sudo", "powerstat", "-R", "1", "3600"]
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc

#################################################################################
##################### main method
if __name__ == "__main__":


    parameter_values = [1000, 5000, 10000]
    layer_values = [1,2,3,4]

    #####power idle and plain
    base_power_measurement()
    mem_file_plain = open(f"results/mlp/basic/powerstat_bench_plain.txt", "w")
    proc = bench_power_measurement(mem_file_plain)  
    
    plain_bench(200000, 10000, 4)
    proc.terminate()
    proc.wait()
    mem_file_plain.close()
    print("done with idle and plain power usage")

    #############accuracy and time
    for i in parameter_values:
        mem_file = open(f"results/mlp/basic/powerstat_bench_{i}p.txt", "w")
        for j in layer_values:
            test(j,i,5, mem_file)
            print(f"done with parameters: {i} and layers: {j}")


    with open(f'results/mlp/basic/data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["parameters", "layers", "mean plain [s]", "std plain", "mean encryped [s]","std encrypted", "relative error", "sample size"])
        for i in range(len(cipher_times)):
            writer.writerow([parameter_amounts[i], layer_amounts[i], round(plain_times[i],6), round(plain_times_std[i],6), round(cipher_times[i],4), 
                             round(cipher_times_std[i],4), round(mean_error[i],4),sample_sizes[i]-2])    
                