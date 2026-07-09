
from time import perf_counter
import csv
import subprocess

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import RandomForestRegressor
     


plain_times = []
plain_times_std = []
mean_error = []
cipher_times = []
cipher_times_std = []
sample_sizes = []
parameter_amounts = []
layer_amounts = []
amount_features = []
amount_estimators = []

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
    

def test(features, estimators, sample_size, mem_file):


    data, train_data, train_result = generateDataset(sample_size, features)
    fhe_model = RandomForestRegressor(n_bits=8, n_estimators=estimators)
    
    fhe_model.fit(train_data, train_result)
    fhe_circuit = fhe_model.compile(train_data)
    fhe_circuit.client.keygen()

    ctimes = []
    ptimes = []
    errors = []

    x = 0
    for i in data:
        begin = perf_counter()
        res_plain = fhe_model.predict(i, fhe="disable")
        end_plain = perf_counter()- begin

        begin = perf_counter()
        res = fhe_model.predict(i, fhe="execute")
        end_enc = perf_counter()- begin

        if x > 3:
            err = abs(res_plain[0]-res[0])/abs(res_plain[0])
            errors.append(err)
            ptimes.append(end_plain)
            ctimes.append(end_enc)
        x += 1

    print(np.mean(errors))
    avgC = np.mean(ctimes)
    std_devC = np.std(ctimes)
    avgP = np.mean(ptimes)
    std_devP = np.std(ptimes)
    
    print(std_devC, ", ", avgC)
    print(std_devP, ", ", avgP)
    if (std_devC*10 > avgC): return test(features, estimators, sample_size*2, mem_file)
    else:
        print("another one")
        plain_times.append(avgP)
        cipher_times.append(avgC)
        plain_times_std.append(std_devP)
        cipher_times_std.append(std_devC)
        mean_error.append(np.mean(errors))
        sample_sizes.append(sample_size)
        amount_features.append(features)
        amount_estimators.append(estimators)

def base_power_measurement():

    cmd = ["sudo", "powerstat", "-R", "1", "60"]

    print("Running base powerstat...")
    with open("results/random_forest/idle_powerstat.txt", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print("Done. Output saved.")

def bench_power_measurement(f):

    cmd = ["sudo", "powerstat", "-R", "1", "3600"]

    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    return proc


if __name__ == "__main__":
    
    estimators = [10, 50, 100, 250, 500]

    base_power_measurement()

    for i in estimators:
        mem_file = open(f"results/forest/powerstat_bench_{i}f.txt", "w")
        for j in estimators:
            proc = bench_power_measurement(mem_file)
            test(i, j , 25+3, mem_file)

            proc.terminate()
            proc.wait()
            print(f"done with number of feature: {i} and percentage of informative: {j}")  

    mem_file.close()
    file = open('results/random_forest/data.csv','a', newline='')
    writer = csv.writer(file)
    writer.writerow(["features", "estimators", "mean plain in s", "std plain", "mean encryped in s","std encrypted", "mean relative error", "sample size"])
        
    for i in range(len(plain_times)):
        writer.writerow([amount_features[i], amount_estimators[i], round(plain_times[i],5), round(plain_times_std[i],5), 
                         round(cipher_times[i],5), round(cipher_times_std[i],5), round(mean_error[i],5), sample_sizes[i]-3])
        file.flush()
            
    file.close()

