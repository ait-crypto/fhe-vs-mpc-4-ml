from time import perf_counter
import csv
import math
import torch
import numpy as np
from torch import nn
import brevitas.nn as qnn
import subprocess

from concrete.ml.torch.compile import compile_brevitas_qat_model
from brevitas.quant import Int8ActPerTensorFloat


N_BITS = 5

class SimpleCNN(nn.Module):
    def __init__(self, params, layers, qidentity_args={"bit_width": N_BITS, "act_quant": Int8ActPerTensorFloat}):
        super(SimpleCNN, self).__init__()

        convIO = 0
        if (layers > 1):
            convIO = abs((-27+math.sqrt(27**2-(4*(layers-1)*9*-params)))/(2*(layers-1)*9))
            convIO = int(round(convIO))
            if (layers == 6 and params == 1000): convIO = 5
        else:
            convIO = int(params/10)

        print("convolution size", convIO)
        layer_list = []
        for i in range(layers):
            if (i == 0):
                layer_list.append(qnn.QuantIdentity(**qidentity_args))
                layer_list.append(qnn.QuantConv2d(3, convIO, kernel_size=3, bias=False, padding=1))
            else:
                layer_list.append(qnn.QuantConv2d(convIO, convIO,kernel_size=3, bias=False, padding=1))

            layer_list.append(qnn.QuantIdentity(**qidentity_args))
            layer_list.append(nn.MaxPool2d(kernel_size=3, stride=1))
            layer_list.append(qnn.QuantReLU(bit_width=qidentity_args["bit_width"]))

        self.all_layers = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.all_layers(x)
        return x




plain_times = []
plain_times_std = []
mean_error = []
cipher_times = []
cipher_times_std = []
sample_sizes = []
parameter_amounts = []
picture_sizes = []



def make_data(sample_size:int, picture_size:int):
    data = []
    for i in range(sample_size):
        data.append(torch.randn((1,3,picture_size, picture_size)))
    return data

def test(sample_size:int, picture_size:int, parameters:int, layers:int, file):
    model = SimpleCNN(parameters, layers)

    fhe_model = compile_brevitas_qat_model(model, torch.randn(1,3,picture_size, picture_size), n_bits=6, rounding_threshold_bits=6)
    print("compilation done!")
    parameter_amounts.append(parameters)
    picture_sizes.append(picture_size)
    
    proc = bench_power_measurement(mem_file)   
    inference(model, fhe_model, sample_size, picture_size)
    proc.terminate()
    proc.wait()
    
def inference(model, fhe_model, sample_size, picture_size):
    test_data = make_data(sample_size, picture_size)

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

        if x > 0:
            res_plain = res_plain.detach().numpy()
            errors.append(np.mean((res_plain - res) ** 2))
            ptimes.append(end_plain)
            ctimes.append(end_enc)
        x += 1

    print(np.average(errors))
    avgC = np.average(ctimes)
    std_devC = np.std(ctimes)
    avgP = np.average(ptimes)
    std_devP = np.std(ptimes)

    print(std_devC, ", ", avgC)
    print(std_devP, ", ", avgP)

    if (std_devC*10 > avgC): return inference(model, fhe_model, int(sample_size*1.5), picture_size)
    else:
        print("another one")
        plain_times.append(avgP)
        cipher_times.append(avgC)
        plain_times_std.append(std_devP)
        cipher_times_std.append(std_devC)
        mean_error.append(np.average(errors))
        sample_sizes.append(sample_size)
    
#################################################################################
#####################simple bench

def plain_bench(sample_size, params, layers, picture_size):
    test_data = make_data(sample_size, picture_size)
    model = SimpleCNN(params, layers).eval()

    for i in test_data:
        model.forward(i)

#################################################################################
##################### power stuff
def base_power_measurement():

    cmd = ["sudo", "powerstat", "-R", "1", "60"]

    print("Running base powerstat...")
    with open(f"results/cnn/idle_powerstat.txt", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print("Done. Output saved.")

def bench_power_measurement(f):

    cmd = ["sudo", "powerstat", "-R", "1", "3600"]
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc

#################################################################################
##################### main method
if __name__ == "__main__":


    sizes = [8,16,32]
    parameters = [1000,5000,10000]
    layers = [3]

    #####power idle and plain
    base_power_measurement()
    mem_file_plain = open(f"results/cnn/powerstat_bench_plain.txt", "w")
    proc = bench_power_measurement(mem_file_plain)  
    
    plain_bench(200000, 50000, 3, 16)
    proc.terminate()
    proc.wait()
    mem_file_plain.close()
    print("done with idle and plain power usage")


    file = open("/results/cnn/data.csv", "a", newline='')
    writer = csv.writer(file)
    writer.writerow(["parameters", "picture size", "mean plain in s", "std plain", "mean encryped in s","std encrypted", "error", "sample size"])
                
    for i in sizes:
        mem_file = open(f"results/mlp/basic/powerstat_bench_{i}pixel.txt", "w")
        for j in parameters:
            test(10+1, i ,j, 3, mem_file)
            writer.writerow([parameter_amounts[-1], picture_sizes[-1], round(plain_times[-1],4), round(plain_times_std[-1],4), round(cipher_times[-1],4), 
                         round(cipher_times_std[-1],4), round(mean_error[-1],4), sample_sizes[-1]-1])   

            file.flush()
            print(f"done with picture size: {i} and number parameters: {j}")

    file.close()


    
