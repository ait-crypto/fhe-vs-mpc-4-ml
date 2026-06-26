import csv
import sys
import numpy as np
from concrete import fhe
from time import perf_counter

plain_times = []
plain_times_std = []
mean_error = []
cipher_times = []
cipher_times_std = []
sample_sizes = []


@fhe.compiler({"x": "encrypted", "y": "encrypted"})
def l1(x,y):
    diff = np.subtract(x,y)
    abso = np.absolute(diff)
    return np.sum(abso)



def make_data(sample_size, vector_size, bit_size):
    min = -(2**(bit_size-1))
    max = (2**(bit_size-1))-1

    data = []
    for i in range(sample_size):
        data.append([
            np.random.randint(min, max, size=vector_size, dtype=int), 
            np.random.randint(min, max, size=vector_size, dtype=int)
        ])

    return data

def test(sample_size, vectors_size):

    data = make_data(sample_size, vectors_size, 8)

    circuit = l1.compile((([-(2**(8-1))+1] * vectors_size,[(2**(8-1))-2]*vectors_size),([(2**(8-1))-2]*vectors_size,[-(2**(8-1))+1] * vectors_size)), loop_parallelize=True)
    print("compilation done")
    circuit.keygen()


    times_plain = []
    times_enc = []
    errors = [] 
    counter = 0
    for i in data:
        begin = perf_counter()
        res_plain = l1(i[0], i[1])
        end_plain = perf_counter()-begin

        x,y = circuit.encrypt(i[0],i[1])
        begin = perf_counter()
        res_enc = circuit.run(x, y)
        end_enc = perf_counter()-begin
        res_dec = circuit.decrypt(res_enc)


        if counter > 3: 
            times_plain.append(end_plain)
            times_enc.append(end_enc)
            errors.append(np.abs(res_plain-res_dec))

            for j in range(100): #to have more plain measurements
                begin = perf_counter()
                res_plain = l1(i[0], i[1])
                times_plain.append(perf_counter()-begin)

        counter += 1

    if np.std(times_enc)*10 > np.mean(times_enc): test(sample_size*2, vectors_size)  
    else: 
        print("another one")
        plain_times.append(np.mean(times_plain))
        cipher_times.append(np.mean(times_enc))
        plain_times_std.append(np.std(times_plain))
        cipher_times_std.append(np.std(times_enc))
        mean_error.append(np.mean(errors))
        sample_sizes.append(sample_size)      


if __name__ == "__main__":

    vector_sizes = int(sys.argv[1])

    warmup = 3
    file = open("results/distance_measures/l1/data.csv", "a", nwline='')
    writer = csv.writer(file)
    writer.writerow(["vector size", "mean plain [s]", "std plain", "mean encryped [s]","std encrypted", "error", "sample size"])
        
    for i in vector_sizes:
        test(20+warmup, i)
        print(f"done with vector size: {i}")

        writer.writerow([i, round(plain_times[-1],4), round(plain_times_std[-1],4), round(cipher_times[-1],4), 
                             round(cipher_times_std[-1],4), round(mean_error[-1],4), sample_sizes[-1]-1]-warmup)   
        file.flush()

    file.close()




    
