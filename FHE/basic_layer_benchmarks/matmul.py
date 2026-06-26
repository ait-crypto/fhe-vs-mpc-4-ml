
import torch
import csv
import subprocess
import numpy as np
from torch import nn
from time import perf_counter
from concrete.ml.torch.compile import compile_torch_model


N_BITS = 8

plain_times = []
plain_times_std = []
mean_error = []
cipher_times = []
cipher_times_std = []
sample_sizes = []

######## one layer model wiht matmul
class MyMatmul(nn.Module):
    def __init__(self,y):
        super(MyMatmul, self).__init__()
        self.y = y

    def forward(self, x):
        x = torch.matmul(x,self.y)
        return x



def make_data(sample_size, mat_size):
    data = []
    for i in range(sample_size):
        data.append(torch.randn((1,mat_size, mat_size)))
    
    return data


def test(mat_size, sample_size, mem_file):
    model = MyMatmul(torch.randn((1,mat_size,mat_size)))
    fhe_model = compile_torch_model(model, torch.randn((1,mat_size, mat_size)), rounding_threshold_bits=8, n_bits=8)
    print("Compilation done")

    proc = bench_power_measurement(mem_file)
    inference(fhe_model, model, sample_size, mat_size)
    proc.terminate()
    proc.wait()


def inference(fhe_model, model, sample_size, mat_size):
    test_data = make_data(sample_size, mat_size)

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
            mse = torch.mean((res_plain - res) ** 2)
            errors.append(mse)
            ptimes.append(end_enc)
            ctimes.append(end_plain)
        x += 1


    avgC = np.mean(ctimes)
    std_devC = np.std(ctimes)
    avgP = np.mean(ptimes)
    std_devP = np.std(ptimes)
    
    print(std_devC, ", ", avgC)
    print(std_devP, ", ", avgP)
    if (std_devC*10 > avgC or std_devP*3 > avgP): return inference(fhe_model, model, sample_size*2, mat_size)
    else:
        print("another one")
        plain_times.append(avgP)
        cipher_times.append(avgC)
        plain_times_std.append(std_devP)
        cipher_times_std.append(std_devC)
        mean_error.append(np.mean(errors))
        sample_sizes.append(sample_size)



############# power measurement methods
def base_power_measurement():

    cmd = ["sudo", "powerstat", "-R", "1", "60"]

    print("Running base powerstat...")
    with open("results/basic_layers/matmul/idle_powerstat_bench.txt", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print("Done. Output saved.")

def bench_power_measurement(f):

    cmd = ["sudo", "powerstat", "-R", "1", "3600"]

    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    return proc


############## main method
if __name__ == "__main__":

    matrix_sizes = [10,50,100]
    
    warmup = 3
    base_power_measurement()
    mem_file = open(f"results/basic_layers/matmul/powerstat_bench.txt", "w")

    for i in matrix_sizes:
        test(i,20+warmup, mem_file)
        print(f"done with matrix size {i}")
    mem_file.close()

    file = open("results/basic_layers/matmul/data.csv", "a", newline='')
    writer = csv.writer(file)
    writer.writerow(["matrix size", "mean plain [s]", "std plain", "mean encryped [s]","std encrypted", "error", "sample size"])
        
    for i in range(len(matrix_sizes)):
        writer.writerow([matrix_sizes[i], round(plain_times[i],5), round(plain_times_std[i],5), round(cipher_times[i],5), 
                         round(cipher_times_std[i],5), round(mean_error[i],5), sample_sizes[i]-warmup])
        file.flush()
        
    file.close()