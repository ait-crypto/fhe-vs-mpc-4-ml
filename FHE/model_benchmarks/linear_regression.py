
from time import perf_counter
import csv
import subprocess
import gc
import sys
import time

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import LinearRegression



plain_times = []
plain_times_std = []
mean_error = []
cipher_times = []
cipher_times_std = []
sample_sizes = []
parameter_amounts = []
layer_amounts = []

####################################################################
############data stuff
def generateDataset(sample_size, features):


    X, y = make_regression(
        n_samples=int(sample_size/0.4),
        n_features=features,
        n_informative=features,
    )

    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    return x_test, x_train, y_train
    

####################################################################
############overall benchmark
def test(features, sample_size, mem_file, bits):

    data, train_data, train_result = generateDataset(sample_size, features)
    fhe_model = LinearRegression(n_bits=int(bits))
    
    fhe_model.fit(train_data, train_result)
    fhe_model.compile(train_data)

    ptimes = []
    ctimes = []
    errors = []
    x = 0 
    proc = bench_power_measurement(mem_file)
    
    for i in data:
        begin = perf_counter()
        res_plain = fhe_model.predict(i, fhe="disable")
        end_plain = perf_counter()- begin

        begin = perf_counter()
        res = fhe_model.predict(i, fhe="execute")
        end_enc = perf_counter()- begin
        

        if (x > 4):
            err = abs(res_plain[0] - res[0])/abs(res_plain[0])
            errors.append(err)
            ptimes.append(end_plain)
            ctimes.append(end_enc)
        x += 1
    
    proc.terminate()
    #print("errors: ", np.mean(errors))
    
    avgC = np.mean(ctimes)
    std_devC = np.std(ctimes)
    avgP = np.mean(ptimes)
    std_devP = np.std(ptimes)
    
    print(std_devC, ", ", avgC)
    print(std_devP, ", ", avgP)
    if (std_devC*10 > avgC): return test(features, sample_size*2, mem_file, bits)
    else:
        print("another one")
        plain_times.append(avgP)
        cipher_times.append(avgC)
        plain_times_std.append(std_devP)
        cipher_times_std.append(std_devC)
        mean_error.append(np.mean(errors))
        sample_sizes.append(sample_size)

####################################################################
############memory benches
def plain_bench(sample_size, features, bits):
    data, train_data, train_result = generateDataset(sample_size, features)

    model = LinearRegression(n_bits=bits)
    model.fit(train_data,train_result)
    model.compile(train_data)

    for i in data:
        model.predict(i, fhe="disable")

####################################################################
############power measurements
def base_power_measurement(bits):

    cmd = ["sudo", "powerstat", "-R", "1", "60"]

    print("Running base powerstat...")
    with open(f"results/linear_regression/normal/{bits}bit/idle_powerstat.txt", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print("Done. Output saved.")

def bench_power_measurement(f):

    cmd = ["sudo", "powerstat", "-R", "1", "3600"]
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc


####################################################################
############main
if __name__ == "__main__":
    
    bits = int(sys.argv[1])
    feature_size = [10, 50, 100, 250, 500]

    ############## power stuff
    base_power_measurement(bits)

    mem_file_plain = open(f"results/linear_regression/normal/{bits}bit/powerstat_bench_plain.txt", "w")
    proc = bench_power_measurement(mem_file_plain)  
    plain_bench(2000000, 500, bits)
    print("done with plaintext powerstat benchmark")

    proc.terminate()
    proc.wait()

    ######################### time/accuracy bench
    mem_file = open(f"results/linear_regression/normal/{bits}bit/powerstat_bench.txt", "w")

    for i in feature_size:
        test(i,3000+5, mem_file, bits)
        print(f"done with number of feature: {i}")
    
    mem_file.close()
    print("done with time and accuracy benchmarks benchmarks")

        
    file = open(f'results/linear_regression/normal/{bits}bit/data.csv','a', newline='')
    writer = csv.writer(file)
    writer.writerow(["features", "mean plain [s]", "std plain", "mean encryped [s]","std encrypted", "relative error", "sample size"])
        
    for i in range(len(feature_size)):
        writer.writerow([feature_size[i], round(plain_times[i],5), round(plain_times_std[i],5), round(cipher_times[i],5), 
                         round(cipher_times_std[i],5), round(mean_error[i],5), sample_sizes[i]-5])
        file.flush()
        
    file.close()