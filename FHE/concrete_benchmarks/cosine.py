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


@fhe.compiler({"x": "encrypted", "y": "encrypted", "normX": "encrypted", "normY": "encrypted"})
def cosine(x,y, normX, normY):
    

    ## for also norm calculation
    #len_x = np.round(fhe.univariate(lambda x: np.sqrt(x))(np.sum(x**2))).astype(np.int16)
    #len_y = np.round(fhe.univariate(lambda x: np.sqrt(x))(np.sum(y**2))).astype(np.int16)
    
    #product = len_x*len_y
    #inverse = fhe.univariate(lambda y: 1/y)(len_x*len_y)
    
    dot = np.dot(x,y)
    mul = normX*normY
    dot_reduced = np.round(fhe.univariate(lambda x: x/100)(dot)).astype(np.int16)
    mul_reduced = np.round(fhe.univariate(lambda x: x/500)(mul)).astype(np.int16)

    result = fhe.multivariate(lambda y,x: int((1-(y/x)*0.2)*1000))(dot_reduced, mul_reduced)
    return result



def make_data(sample_size, vector_size, bit_size):
    min = -(2**(bit_size-1))
    max = (2**(bit_size-1))-1

    data = []

    for i in range(sample_size):
        x = np.random.randint(min, max, size=vector_size, dtype=int)
        y = np.random.randint(min, max, size=vector_size, dtype=int)
        data.append([
            x,y, round(np.linalg.norm(x)),round(np.linalg.norm(y))
        ])

    return data

def test(sample_size, vectors_size):

    data = make_data(sample_size, vectors_size, 5)

    circuit = cosine.compile(((data[0][0], data[0][1], data[0][2], data[0][3]),), )
    circuit.keygen()
    print("compilation finished")

    times_plain = []
    times_enc = []
    errors = [] 
    counter = 0

    for i in data:
        begin = perf_counter()
        res_plain = cosine(i[0],i[1],i[2],i[3])
        end_plain = perf_counter()-begin

        x,y,normx,normy = circuit.encrypt(i[0],i[1],i[2],i[3])
        
        begin = perf_counter()
        res_enc = circuit.run(x, y,normx,normy)
        end_enc = perf_counter()-begin
        res_dec = circuit.decrypt(res_enc)


        if counter > 1: 
            times_plain.append(end_plain)
            times_enc.append(end_enc)
            errors.append(np.abs(res_plain- res_dec))

            for j in range(100): #to have more plain measurements
                begin = perf_counter()
                res_plain = cosine(i[0], i[1],i[2], i[3])
                times_plain.append(perf_counter()-begin)

        print(f"one done {end_enc}")

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

    file = open("results/distance_measures/cosine/data.csv", "a", newline='')
    writer = csv.writer(file)
    writer.writerow(["vector size", "mean plain [s]", "std plain", "mean encryped [s]","std encrypted", "error", "sample size"])

    for i in vector_sizes:
        test(4+1, i)
        print(f"done with vector size: {i}")
        writer.writerow([i, round(plain_times[-1],4), round(plain_times_std[-1],5), round(cipher_times[-1],4), 
                             round(cipher_times_std[-1],4), round(mean_error[-1],4), sample_sizes[-1]-1]-1)   
        
        file.flush()
    
    file.close()




    
