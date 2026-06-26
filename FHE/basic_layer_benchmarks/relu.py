
import torch
import csv
import subprocess
import numpy as np
import brevitas.nn as qnn
from time import perf_counter
from brevitas.quant import Int8ActPerTensorFloat
from concrete.ml.torch.compile import compile_brevitas_qat_model
from torch import nn



plain_times = []
plain_times_std = []
mean_error = []
cipher_times = []
cipher_times_std = []
sample_sizes = []


N_BITS = 8

######## one layer model with relu activation function
class MyRelu(nn.Module):
    def __init__(self,qidentity_args={"bit_width": N_BITS, "act_quant": Int8ActPerTensorFloat}):
        super(MyRelu, self).__init__()
        self.ident = qnn.QuantIdentity(**qidentity_args)
        self.relu = qnn.QuantReLU()
    def forward(self, x):
        x = self.ident(x)
        x = self.relu(x)
        return x



def make_data(sample_size, vector_size):
    data = []
    for i in range(sample_size):
        data.append(torch.randn((1,vector_size)))
    
    return data

def test(vec_size, sample_size, mem_file):
    model = MyRelu()
    fhe_model = compile_brevitas_qat_model(model, torch.randn((1,vec_size)), rounding_threshold_bits=8, n_bits=8)
    print("Compilation done")

    proc = bench_power_measurement(mem_file)
    inference(fhe_model, model, sample_size, vec_size)
    proc.terminate()
    proc.wait()



def inference(fhe_model, model, sample_size, vec_size):
    test_data = make_data(sample_size, vec_size)

    ctimes= []
    ptimes= []
    errors = []

    x = 0
    for i in test_data:
        begin = perf_counter()
        res = fhe_model.forward(i.numpy(), fhe="execute")
        end_plain = perf_counter()-begin

        begin = perf_counter()
        res_plain = model.forward(i)
        end_enc = perf_counter()-begin
        
        if x > 3:
            res_plain = res_plain.detach().numpy()
            err = np.mean((res_plain - res) ** 2)
            errors.append(err)
            ptimes.append(end_enc)
            ctimes.append(end_plain)
        x += 1


    avgC = np.mean(ctimes)
    std_devC = np.std(ctimes)
    avgP = np.mean(ptimes)
    std_devP = np.std(ptimes)
    print(np.mean(errors))
    
    print(std_devC, ", ", avgC)
    print(std_devP, ", ", avgP)
    if (std_devC*10 > avgC or std_devP*3 > avgP): return inference(fhe_model, model, sample_size*2, vec_size)
    else:
        print("another one")
        plain_times.append(avgP)
        cipher_times.append(avgC)
        plain_times_std.append(std_devP)
        cipher_times_std.append(std_devC)
        mean_error.append(np.mean(errors))
        sample_sizes.append(sample_size)

############ powerstat benchmark methods
def base_power_measurement():

    cmd = ["sudo", "powerstat", "-R", "1", "60"]

    print("Running base powerstat...")
    with open("results/basic_layers/relu/idle_powerstat_benc.txt", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print("Done. Output saved.")

def bench_power_measurement(f):

    cmd = ["sudo", "powerstat", "-R", "1", "3600"]

    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    return proc


############ main
if __name__ == "__main__":

    vector_sizes = [8,16,32]
    warmup = 3

    base_power_measurement()
    mem_file = open(f"results/basic_layers/relu/powerstat_bench.txt", "w")

    for i in vector_sizes:
        test(i,20+warmup, mem_file)
        print(f"done with vector size {i}")
    mem_file.close()

    file = open("results/basic_layers/relu/data.csv", "a", newline='')
    writer = csv.writer(file)
    writer.writerow(["vector size", "mean plain [s]", "std plain", "mean encryped [s]","std encrypted", "error", "sample size"])
        
    for i in range(len(vector_sizes)):
        writer.writerow([vector_sizes[i], round(plain_times[i],5), round(plain_times_std[i],5), round(cipher_times[i],5), 
                         round(cipher_times_std[i],5), round(mean_error[i],5), sample_sizes[i]-warmup])
        file.flush()
        
    file.close()