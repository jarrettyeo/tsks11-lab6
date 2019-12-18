# validate
## y, n
# jarye
## n, n
# task2
## 2, n, y
# task2 (improved)
## 2, n, y, n, n

import numpy as np
from matplotlib import pyplot as plt 

from math import sqrt

from yaspin import yaspin
import pickle
import itertools

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

if validate == '2':
    run_improved = input("Run improved? (y/n) ") or 'n'
    assert run_improved in ['y','n']

    rerun_distance_matrix = input("Rerun distance_matrix? (y/n) ") or 'n'
    assert rerun_distance_matrix in ['y','n']

    rerun_improved_matrix = input("Rerun improved_matrix? (y/n) ") or 'n'
    assert rerun_improved_matrix in ['y','n']

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
            pickle.dump(bias, open("BASE-validate-bias.p", "wb"))
        elif validate == '2':
            pickle.dump(bias, open("BASE-task2-bias.p", "wb"))
        else:
            pickle.dump(bias, open("BASE-jarye-bias.p", "wb"))

else:
    if validate == 'y':
        bias = pickle.load(open("BASE-validate-bias.p", "rb"))
    if validate == '2':
        bias = pickle.load(open("BASE-task2-bias.p", "rb"))        
    else:
        bias = pickle.load(open("BASE-jarye-bias.p", "rb"))

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
plt.xlabel("Absolute error")
plt.ylabel("Count")
if validate == 'y':
    plt.title("Baseline Validate Absolute Errors")
    plt.savefig("BASE-validate.png")
elif validate == '2':
    plt.title("Baseline Task 2 Absolute Errors")
    plt.savefig("BASE-task2.png")    
else:
    plt.title("Baseline jarye Absolute Errors")
    plt.savefig("BASE-jarye.png")
plt.close()
error_dist = [x for x in hist if x > 0]
print(f"{error_dist=}")

# =========================================================================

if run_improved == 'y' and validate == '2':

    U_MIN = 100
    NEIGHBOURS_NUMBER_LIST = [100]
    PROPORTION_OF_PREDICT_VS_SIM_LIST = [(1,0.5)]

    difference_matrix = np.full((max_user, max_user), np.nan)
    u_indexes, m_indexes = np.nonzero(training_matrix) # if rating exists
    for u_index, m_index in zip(u_indexes, m_indexes):
        difference_matrix[u_index, m_index] = training_matrix[u_index, m_index] - training_matrix_predicted[u_index, m_index]

    if rerun_distance_matrix == 'y':
        distance_matrix = np.zeros((max_movie, max_movie))
        for m1 in range(max_movie):
            print('Creating distance_matrix for {m1}/{max_movie}...'.format(
                m1=m1+1, max_movie=max_movie
            ))
            for m2 in range(m1+1, max_movie):
                movie_a, movie_b = zip(*[(difference_matrix[u_index, m1], difference_matrix[u_index, m2]) for u_index in range(max_user) if not np.nan(difference_matrix[u_index, m1]) and not np.nan(difference_matrix[u_index, m2]) ])
                distance_matrix[m1, m2] = distance_matrix[m2, m1] = distance.cosine(movie_a, movie_b) if len(movie_a) > U_MIN else 0.0
        pickle.dump(distance_matrix, open("IMPROVED-task2-distance_matrix-U_MIN-{U_MIN}.p".format(U_MIN=str(U_MIN)), "wb"))
    else:
        distance_matrix = pickle.load(open("IMPROVED-task2-distance_matrix-U_MIN-{U_MIN}.p".format(U_MIN=str(U_MIN)), "rb"))

    def trial_and_error(NEIGHBOURS_NUMBER, PROPORTION_OF_PREDICT, PROPORTION_OF_SIM):

        if rerun_improved_matrix == 'y':

            improved_matrix = np.full([max_user, max_movie], np.nan)
            print('\n')
            for m_index in range(max_movie):
                print('Finding similarity for {m_index}/{max_movie}...'.format(
                    m_index=m_index+1, max_movie=max_movie
                ))
                neighbours = np.argsort([abs(x) for x in distance_matrix[m_index]])[::-1][:NEIGHBOURS_NUMBER]
                for u_index in range(max_user):

                    neighbourhood_denominator = sum([abs(distance_matrix[m_index, neighbour]) for neighbour in neighbours if not training_matrix[u_index, neighbour] == 0.0])

                    if not neighbourhood_denominator == 0:
                        neighbourhood_numberator = sum([distance_matrix[m_index, neighbour] * difference_matrix[u_index, neighbour] for neighbour in neighbours if not training_matrix[u_index, neighbour] == 0.0])
                        similarity_score = neighbourhood_numberator / neighbourhood_denominator
                    elif u_index < U_MIN * sum(PROPORTION_OF_PREDICT_VS_SIM_LIST[0]):
                        similarity_score = np.nan
                    else:
                        similarity_score = 0.0

                    improved_matrix[u_index, m_index] = training_matrix_predicted[u_index, m_index] + similarity_score if u_index > U_MIN else np.nan

            improved_matrix = np.clip(improved_matrix, 1, 5)

            pickle.dump(improved_matrix, open("IMPROVED-task2-improved_matrix-U_MIN-{U_MIN}.p".format(U_MIN=str(U_MIN)), "wb"))
        else:
            improved_matrix = pickle.load(open("IMPROVED-task2-improved_matrix-U_MIN-{U_MIN}.p".format(U_MIN=str(U_MIN)), "rb"))

        rmse_test_improved = 0
        histogram_data = []
        u_indexes, m_indexes = range(max_user), range(max_movie)
        for u_index, m_index in zip(u_indexes, m_indexes):
            r_um_predicted = improved_matrix[u_index, m_index]
            r_um = test_matrix[u_index, m_index]
            abs_error = abs(r_um - r_um_predicted) if not np.isnan(r_um_predicted) else 0 # TODO
            histogram_data.append(abs_error)
            rmse_test_improved += abs_error ** 2 if not np.isnan(r_um_predicted) else 0 # TODO
        rmse_test_improved = sqrt(rmse_test_improved / test_size)
        rmse_test_improved = round(rmse_test_improved,3)
        print(f"{rmse_test_improved=}")

        assert rmse_test_improved == 0.903

        hist, bins = np.histogram(histogram_data, bins=[0,1,2,3,4,5])
        plt.figure(1)
        plt.bar([0.5, 1.5, 2.5, 3.5, 4.5], hist)
        plt.title("Improved Task 2 Absolute Errors")
        plt.xlabel("Absolute Error")
        plt.ylabel("Count")
        plt.savefig("IMPROVED-task2.png")
        plt.close()
        error_dist_improved = [x for x in hist if x > 0]
        print(f"{error_dist_improved=}")    

    for NEIGHBOURS_NUMBER, PROPORTION_OF_PREDICT_VS_SIM in itertools.product(NEIGHBOURS_NUMBER_LIST, PROPORTION_OF_PREDICT_VS_SIM_LIST):
        PROPORTION_OF_PREDICT, PROPORTION_OF_SIM = PROPORTION_OF_PREDICT_VS_SIM
        print(f"{U_MIN=}; {NEIGHBOURS_NUMBER=}; {PROPORTION_OF_PREDICT_VS_SIM=}; ", end="")
        trial_and_error(NEIGHBOURS_NUMBER, PROPORTION_OF_PREDICT, PROPORTION_OF_SIM)