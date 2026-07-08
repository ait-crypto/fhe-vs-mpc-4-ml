import sys
import csv
import numpy as np
from concrete.ml.sklearn import NeuralNetRegressor
from  time import perf_counter

from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

     

plain_times = []
plain_accuracy = []
accuracy = [] 
mean_error = []
max_error = []
cipher_times = []
cipher_times_std = []
sample_sizes = []
layer_amounts = []
trained_epochs = []
min = []
max = []

def make_data(sample_size):
    X, y = make_regression(
        n_samples=int(sample_size/0.4),
        n_features=10,
        n_informative=5,
    )

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Scikit-Learn and Concrete ML neural networks only handle float32 input values
    X_train, X_test = X_train.astype("float32"), X_test.astype("float32")
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    return X_train, X_test, y_train, y_test




def test(layers, epochs, sample_size, bits):


    train_data, data, train_result, data_result  = make_data(sample_size)    
    max.append(np.max(data_result))
    min.append(np.min(data_result))
    
    params = {
        "module__n_layers": layers,
        "module__activation_function": nn.ReLU,
        "module__n_w_bits": bits,
        "module__n_a_bits": bits,
        "max_epochs":epochs,
        "verbose": 0,
    }
    
    model = NeuralNetRegressor(**params)
    model, sklearn_model = model.fit_benchmark(X=train_data, y=train_result)


    y_pred_sklearn = sklearn_model.predict(data)

    #### plain accuracy
    sklearn_accuracy = r2_score(data_result, y_pred_sklearn) * 100
    plain_accuracy.append(sklearn_accuracy);
    print(f"The test accuracy of the trained scikit-learn model is {sklearn_accuracy:.2f}%")

    ##### plain speed
    begin = perf_counter()
    for i in range(10):
        y_pred_sklearn = sklearn_model.predict(data)
    plain_times.append((perf_counter()-begin)/ len(data)/10)


    fhe_circuit = model.compile(train_data)
    print("Generating a key for a " f"{fhe_circuit.graph.maximum_integer_bit_width()}-bit circuit")
    time_begin = perf_counter()
    fhe_circuit.client.keygen(force=True)
    print(f"Key generation time: {perf_counter() - time_begin:.2f} seconds")

    x = 0 

    fhe_predictions = []
    fhe_times = []
    #### fhe tests
    for x in tqdm(data):
        time_begin = perf_counter()
        y_ = model.predict(np.array([x]), fhe="execute")[0]
        fhe_times.append(perf_counter()- time_begin)
        fhe_predictions.append(y_)

    ### fhe times
    cipher_times.append(np.mean(fhe_times))
    cipher_times_std.append(np.std(fhe_times))
    
    ### fhe accuracy
    fhe_accuracy = r2_score(data_result, fhe_predictions) * 100
    accuracy.append(fhe_accuracy)
    print(f"The test accuracy of the trained fhe model is {fhe_accuracy:.2f}%")


    absolute_error = np.abs(y_pred_sklearn - fhe_predictions)
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
    layers = [1,2,3,4]

    epochs = 2000
    samples = 50
    for i in layers:
        test(i, epochs, samples, bits)
        print(f"done with number of layers: {i}")
    print("done with time and accuracy benchmarks benchmarks")
    
    file = open(f'results/mlp/trained/{bits}bit/data.csv','a', newline='')
    writer = csv.writer(file)
    writer.writerow(["layers", "sample size", "epochs", "mean plain [s]", "mean encryped [s]","std encrypted", 
                     "plain r2", "encrypted r2", "MAE", "max absolut error", "min", "max"])
        
    for i in range(len(layers)):
        writer.writerow([layers[i], samples, epochs, round(plain_times[i],5), round(cipher_times[i],5), round(cipher_times_std[i],5), 
                         plain_accuracy[i], accuracy[i], mean_error[i], max_error[i], min[i], max[i]])  
        file.flush()
        
    file.close()