# validate
## y, n
# jarye
## n, n
# task2
## 2, n, y
# task2 (improved)
## 2, n, y, n

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
plt.title("Baseline Absolute Errors")
plt.xlabel("Absolute error")
plt.ylabel("Count")
if validate == 'y':
    plt.savefig("BASE-validate.png")
elif validate == '2':
    plt.savefig("BASE-task2.png")    
else:
    plt.savefig("BASE-jarye.png")
error_dist = [x for x in hist if x > 0]
print(f"{error_dist=}")

# =========================================================================

if run_improved == 'y' and validate == '2':

    U_MIN = 100
    NEIGHBOURS_NUMBER_LIST = [100] # min 1000
    PROPORTION_OF_PREDICT_VS_SIM_LIST = [(1,0.5)]

    # fill with special values
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
        pickle.dump(distance_matrix, open("task2-distance_matrix-U_MIN-{U_MIN}.p".format(U_MIN=str(U_MIN)), "wb"))
    else:
        distance_matrix = pickle.load(open("task2-distance_matrix-U_MIN-{U_MIN}.p".format(U_MIN=str(U_MIN)), "rb"))

    def find_best_neighbours(movie, n):
        # returns indices of 2 closes neighbours
        return np.argsort([abs(x) for x in distance_matrix[movie]])[::-1][:n]

    def get_similarity(training_data, user, movie, neighbours):
        sim_denominator = 0.0
        sim_numerator = 0.0
        for neighbour in neighbours:
            if training_data[user, neighbour] != 0:
                sim_numerator += (
                    distance_matrix[movie, neighbour]
                    * difference_matrix[user, neighbour]
                )
                sim_denominator += abs(distance_matrix[movie, neighbour])
        if sim_denominator != 0:
            return sim_numerator * 1.0 / sim_denominator
        elif user < U_MIN * sum(PROPORTION_OF_PREDICT_VS_SIM_LIST[0]):
            return np.nan
        else:
            return 0.0

    def trial_and_error(NEIGHBOURS_NUMBER, PROPORTION_OF_PREDICT, PROPORTION_OF_SIM):
        improved_matrix = np.full([max_user, max_movie], np.nan)
        sim_data = []
        
        print('\n')
        for movie in range(max_movie):
            
            print('Finding similarity for {movie}/{max_movie}...'.format(
                movie=movie+1, max_movie=max_movie
            ))
            # sim = 0.0
            # choose 2 closest neighbours, based on distance matrix
            neighbours = find_best_neighbours(movie, NEIGHBOURS_NUMBER)
            for user in range(max_user):
                similarity = get_similarity(training_matrix, user, movie, neighbours)
                # temp_val = training_matrix_predicted[user, movie] + similarity / 10.0
                # temp_val = PROPORTION_OF_PREDICT * training_matrix_predicted[user, movie] + PROPORTION_OF_SIM * similarity
                temp_val = training_matrix_predicted[user, movie] + similarity
                # sim += similarity
                # crop values
                if temp_val < 1.0:
                    temp_val = 1.0
                elif temp_val > 5.0:
                    temp_val = 5.0
                sim_data.append(similarity)
                improved_matrix[user, movie] = temp_val if user > U_MIN else np.nan # TODO

        #  ---

        # self.rmse_test_improved = self.get_rmse_test(test_data, self.improved_matrix)

        rmse_test_improved = 0
        histogram_data = []
        # u_indexes, m_indexes = np.nonzero(test_matrix)
        u_indexes, m_indexes = range(max_user), range(max_movie) # TODO potential - loop through full array, not just nonzero
        for u_index, m_index in zip(u_indexes, m_indexes):
            r_um_predicted = improved_matrix[u_index, m_index]
            r_um = test_matrix[u_index, m_index]
            # abs_error = abs(r_um - int(r_um_predicted)) * PROPORTION_OF_PREDICT if not np.isnan(r_um_predicted) else 0
            # abs_error = abs(r_um - r_um_predicted) if not np.isnan(r_um_predicted) else 0 # TODO
            abs_error = abs(r_um - r_um_predicted) if not np.isnan(r_um_predicted) else 0 # TODO
            histogram_data.append(abs_error)
            rmse_test_improved += abs_error ** 2 if not np.isnan(r_um_predicted) else 0 # TODO
        rmse_test_improved = sqrt(rmse_test_improved / test_size)
        rmse_test_improved = round(rmse_test_improved,3)
        print(f"{rmse_test_improved=}")

        assert rmse_test_improved == 0.903

        #  ---

        # plt.figure(1)
        # _ = plt.hist(sim_data, bins=100)
        # plt.show()
        # # plt.close()

        # ---
        hist, bins = np.histogram(histogram_data, bins=[0,1,2,3,4,5])
        plt.figure(1)
        plt.bar([0.5, 1.5, 2.5, 3.5, 4.5], hist)
        plt.title("Baseline Absolute Errors")
        plt.xlabel("Absolute error")
        plt.ylabel("Count")
        plt.savefig("task2-improved_abs_error.png")
        plt.close()
        error_dist_improved = [x for x in hist if x > 0]
        print(f"{error_dist_improved=}")    

    # for a,b in itertools.product(a, b):
    for NEIGHBOURS_NUMBER, PROPORTION_OF_PREDICT_VS_SIM in itertools.product(NEIGHBOURS_NUMBER_LIST, PROPORTION_OF_PREDICT_VS_SIM_LIST):
        PROPORTION_OF_PREDICT, PROPORTION_OF_SIM = PROPORTION_OF_PREDICT_VS_SIM
        print(f"{U_MIN=}; {NEIGHBOURS_NUMBER=}; {PROPORTION_OF_PREDICT_VS_SIM=}; ", end="")
        trial_and_error(NEIGHBOURS_NUMBER, PROPORTION_OF_PREDICT, PROPORTION_OF_SIM)