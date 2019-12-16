import numpy as np
from matplotlib import pyplot as plt 

from math import sqrt

from yaspin import yaspin
import pickle

# BASE
## validate
### rmse_training = 0.886
### rmse_test = 0.898
## jarye
### rmse_training = 0.892
### rmse_test = 0.918
## task2
### rmse_training = 0.896
### rmse_test = 0.910

validate = input("Validate? (y/n/2) ") or 'n'
assert validate in ['y','n','2']

rerun_bias = input("Rerun bias? (y/n) ") or 'n'
assert rerun_bias in ['y','n']

if validate == 'y':
    training_filename = "validate_code_netflix/validate_code.training"
    test_filename = "validate_code_netflix/validate_code.test"
elif validate == '2':
    training_filename = "training/task---2.training"
    test_filename = "test/task---2.test"
else:
    training_filename = "training/jarettye.training"
    test_filename = "test/jarettye.test"

def generate_matrix(filename):
    print("\n")
    edge_list = np.loadtxt(filename, delimiter=',')

    users, movies, ratings = edge_list[:,0], edge_list[:,1], edge_list[:,2]
    max_user = int(max(users))
    max_movie = int(max(movies))

    A = np.zeros((max_user, max_movie))
    for u, m, r in zip(users, movies, ratings):
        A[int(u)-1, int(m)-1] = int(r) # -1
    
    return A, len(ratings), max_user, max_movie

training_matrix, training_size, training_max_user, training_max_movie = generate_matrix(training_filename)
test_matrix, test_size, test_max_user, test_max_movie = generate_matrix(test_filename)

max_user = max(training_max_user, test_max_user)
max_movie = max(training_max_movie, test_max_movie)

print(f"{training_size=}")
print(f"{test_size=}")
print(f"{max_user=}")
print(f"{max_movie=}")

training_indices = np.nonzero(training_matrix) # find the indices of an array, where a condition is True
training_indices = np.transpose(training_indices)

r_sum = np.sum(training_matrix)
r_avg = r_sum / training_size

print("\n")

if rerun_bias == 'y':

    A = np.zeros((training_size, max_user + max_movie)) # M x N matrix
    r = np.zeros(training_size) # 1 x N vector with r
    i = 0
    
    for row in training_indices:
        A[i, int(row[0])] = 1
        A[i, int(row[1]) + max_user] = 1
        r[i] = training_matrix[row[0], row[1]] - r_avg
        i += 1

    print("\n")
    print(f"{A=}")
    print(f"{r=}")

    with yaspin(text = "Running least-squares solution to linear matrix equation...").white.bold.shark.on_blue as sp:
        bias, _, _, _ = np.linalg.lstsq(A, r, rcond=1.e-3)
        print(f"\n{bias=}")
        
        if validate == 'y':
            pickle.dump(bias, open("validate-bias.p", "wb"))
        elif validate == '2':
            pickle.dump(bias, open("2-bias.p", "wb"))
        else:
            pickle.dump(bias, open("bias.p", "wb"))

else:
    if validate == 'y':
        bias = pickle.load(open("validate-bias.p", "rb"))
    if validate == '2':
        bias = pickle.load(open("2-bias.p", "rb"))        
    else:
        bias = pickle.load(open("bias.p", "rb"))

training_matrix_predicted = np.zeros((max_user, max_movie))

with yaspin(text = "Creating Training Matrix Predicted...").white.bold.shark.on_blue as sp:
    for (u_index,m_index), _ in np.ndenumerate(training_matrix_predicted):
        r_um_predicted = r_avg + bias[u_index] + bias[m_index + max_user]
        training_matrix_predicted[u_index, m_index] = r_um_predicted
    training_matrix_predicted = np.clip(training_matrix_predicted, 1, 5)

    print("\n")
    print(f"{training_matrix_predicted=}")
    print(f"{r_avg=}")

rmse_training = 0
u_indexes, m_indexes = np.nonzero(training_matrix)
for u_index, m_index in zip(u_indexes, m_indexes):
    r_um_predicted = training_matrix_predicted[u_index, m_index]
    r_um = training_matrix[u_index, m_index]
    rmse_training += (r_um - r_um_predicted) ** 2
rmse_training = sqrt(rmse_training / training_size)
rmse_training = round(rmse_training,3)
print(f"{rmse_training=}")

rmse_test = 0
histogram_data = []
u_indexes, m_indexes = np.nonzero(test_matrix)
for u_index, m_index in zip(u_indexes, m_indexes):
    r_um_predicted = training_matrix_predicted[u_index, m_index]
    r_um = test_matrix[u_index, m_index]
    abs_error = abs(r_um - int(r_um_predicted))
    histogram_data.append(abs_error)
    rmse_test += (r_um - r_um_predicted) ** 2
rmse_test = sqrt(rmse_test / test_size)
rmse_test = round(rmse_test,3)
print(f"{rmse_test=}")

if validate == 'y':
    assert rmse_training == 0.886
    assert rmse_test == 0.898

hist, bins = np.histogram(histogram_data, bins=[0,1,2,3,4,5])
plt.bar([0.5, 1.5, 2.5, 3.5, 4.5], hist)
plt.title("Baseline Absolute Errors")
plt.xlabel("Absolute error")
plt.ylabel("Count")
if validate == 'y':
    plt.savefig("validate-baseline_abs_error.png")
elif validate == '2':
    plt.savefig("2-baseline_abs_error.png")    
else:
    plt.savefig("baseline_abs_error.png")
error_dist = [x for x in hist if x > 0]
print(f"{error_dist=}")