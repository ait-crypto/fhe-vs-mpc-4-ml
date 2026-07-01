from  time import perf_counter

import numpy as np
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

import sys
import csv
from concrete.ml.sklearn import LinearRegression
from sklearn.linear_model import LinearRegression as LinearSKLRegression


plain_times = []
plain_accuracy = []
accuracy = [] 
mean_error = []
max_error = []
cipher_times = []
cipher_times_std = []
sample_sizes = []
trained_epochs = []
min = []
max = []

def make_data(sample_size, features):
    X, y = make_regression(
        n_samples=int(sample_size/0.4),
        n_features=features,
        n_informative=features,
    )

    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    return x_train, x_test, y_train, y_test




def test(features, sample_size, bits):


    train_data, data, train_result, data_result  = make_data(sample_size, features)    
    max.append(np.max(data_result))
    min.append(np.min(data_result))
    
    
    model = LinearRegression(n_bits=bits)
    model= model.fit(X=train_data, y=train_result)

    sklearn_lr = LinearSKLRegression()
    sklearn_lr.fit(train_data, train_result)


    ##### plain speed
    begin = perf_counter()
    for i in range(10):
        y_pred = sklearn_lr.predict(data)
    plain_times.append((perf_counter()-begin)/ len(data)/10)

    #### plain accuracy
    sklearn_accuracy = r2_score(data_result, y_pred) * 100
    plain_accuracy.append(sklearn_accuracy);
    print(f"The test accuracy of the trained scikit-learn model is {sklearn_accuracy:.2f}%")
    

    fhe_circuit = model.compile(train_data)
    print("Generating a key for a " f"{fhe_circuit.graph.maximum_integer_bit_width()}-bit circuit")
    time_begin = perf_counter()
    fhe_circuit.client.keygen(force=True)
    print(f"Key generation time: {perf_counter() - time_begin:.2f} seconds")

    fhe_results = []
    fhe_times = []
    #### fhe tests
    for x in tqdm(data):
        time_begin = perf_counter()
        y_ = model.predict(np.array([x]), fhe="execute")[0]
        fhe_times.append(perf_counter()- time_begin)
        fhe_results.append(y_)   
    
    #### fhe times
    cipher_times.append(np.mean(fhe_times))
    cipher_times_std.append(np.std(fhe_times)) 
    
    #### fhe accuracy
    fhe_accuracy = r2_score(data_result, fhe_results) * 100
    accuracy.append(fhe_accuracy)
    print(f"The test accuracy of the trained fhe model is {fhe_accuracy:.2f}%")

    #### fhe errors
    absolute_error = np.abs(y_pred - fhe_results)
    print(f"fhe accuracy: {fhe_accuracy} ")
    print(f"MEA: {np.mean(absolute_error)} ")
    print(f"max absolute error: {np.max(absolute_error)} ")
    mean_error.append(np.mean(absolute_error))
    max_error.append(np.max(absolute_error))
    sample_sizes.append(sample_size)


####################################################################
############main
if __name__ == "__main__":
    
    bits = int(sys.argv[1])
    features = [10,50,100,250,500]

    samples = 3000
    for i in features:
        test(i, samples, bits)
        print(f"done with number of feature: {i}")
    print("done with time and accuracy benchmarks benchmarks")

    file = open(f'results/linear_regression/trained/{bits}bit/data.csv','a', newline='')
    writer = csv.writer(file)
    writer.writerow(["layers", "sample size", "mean plain [s]", "plain r2", "mean encryped [s]","std encrypted", 
                     "encrypted r2", "MAE", "max absolut error", "min", "max"]) 
        
    for i in range(len(features)):
        writer.writerow([features[i], samples, round(plain_times[i],5), plain_accuracy[i], round(cipher_times[i],5), 
                         round(cipher_times_std[i],5), accuracy[i], mean_error[i], max_error[i], min[i], max[i]]) 
        file.flush()
        
    file.close()