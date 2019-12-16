import numpy as np
from matplotlib import pyplot as plt 
from scipy.spatial import distance

from math import sqrt

from yaspin import yaspin
import pickle
import itertools

task_no = input("Task no? (1/2) ")
assert task_no in ['1','2']

U_MIN = 100
NEIGHBOURS_NUMBER_LIST = [100] # min 1000
PROPORTION_OF_PREDICT_VS_SIM_LIST = [(0.7,1)]

# Task 2 Baseline
# rmse_training=0.896
# rmse_test=0.91
# error_dist_baseline=[6739, 10622, 2817, 168, 8]

# U_MIN=50; NEIGHBOURS_NUMBER=5000; PROPORTION_OF_PREDICT_VS_SIM=(1, 1); rmse_test_improved=1.044
# U_MIN=0; NEIGHBOURS_NUMBER=1000; PROPORTION_OF_PREDICT_VS_SIM=(1, 1); rmse_test_improved=1.045
# U_MIN=0; NEIGHBOURS_NUMBER=100; PROPORTION_OF_PREDICT_VS_SIM=(1, 1); rmse_test_improved=1.15
# U_MIN=0; NEIGHBOURS_NUMBER=10; PROPORTION_OF_PREDICT_VS_SIM=(1, 1); rmse_test_improved=1.124
# U_MIN=0; NEIGHBOURS_NUMBER=1; PROPORTION_OF_PREDICT_VS_SIM=(1, 1); rmse_test_improved=1.079
# U_MIN=1000; NEIGHBOURS_NUMBER=1000; PROPORTION_OF_PREDICT_VS_SIM=(1, 1); rmse_test_improved=1.075
# U_MIN=100; NEIGHBOURS_NUMBER=1000; PROPORTION_OF_PREDICT_VS_SIM=(1, 1); rmse_test_improved=1.08
# U_MIN=100; NEIGHBOURS_NUMBER=1000; PROPORTION_OF_PREDICT_VS_SIM=(0.8, 0.2); rmse_test_improved=1.16

rerun_bias = input("Rerun bias? (y/n) ") or 'n'
assert rerun_bias in ['y','n']

rerun_distance_matrix = input("Rerun distance_matrix? (y/n) ") or 'n'
assert rerun_distance_matrix in ['y','n']

if task_no == '1':
    training_filename = "training/jarettye.training"
    test_filename = "test/jarettye.test"
elif task_no == '2':
    training_filename = "training/task---2.training"
    test_filename = "test/task---2.test"

# training_proportion = 0.9

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

# # self.training_size = self.get_training_size()

# def get_training_size(self):
#     actual_size = int(self.ratings_number * TRAINING_FRACTION)
#     return actual_size if actual_size > 0 else 1

# total_size = len(ratings)
# training_size = int(total_size * training_proportion)
# test_size = len(test_matrix)

# ---

# # self.training_indices = np.transpose(np.nonzero(training_data))

training_indices = np.nonzero(training_matrix) # find the indices of an array, where a condition is True
training_indices = np.transpose(training_indices)

# ---

# # self.matrix_A, self.vector_y = self.construct_a_matrix(training_data)

# def construct_a_matrix(self, matrix_R):
#     # select training set of matrix R
#     np.random.shuffle(self.training_indices)
#     self.training_indices = self.training_indices[: self.training_size]
#     # calculate average rating of training set
#     sum = 0
#     for row in self.training_indices:
#         sum += matrix_R[row[0], row[1]]
#     self.r_avg = sum * 1.0 / self.training_size
#     # construct the matrix A
#     A = np.zeros((self.training_size, self.users + self.movies))
#     r = np.zeros(self.training_size)
#     i = 0
#     for row in self.training_indices:
#         A[i, int(row[0])] = 1
#         A[i, int(row[1]) + self.users] = 1
#         r[i] = matrix_R[row[0], row[1]] - self.r_avg
#         i += 1
#     return A, r

# # select training set of matrix R
# np.random.shuffle(training_indices)
# training_indices = training_indices[: training_size]
# # calculate average rating of training set
# sum = 0
# for row in training_indices:
#     sum += training_matrix[row[0], row[1]]

r_sum = np.sum(training_matrix)
# for row in training_matrix:
#     r_sum += training_matrix[row[0], row[1]]

# r_avg = sum * 1.0 / training_size
r_avg = r_sum / training_size

# ---

# # self.bias = self.get_bias(self.matrix_A, self.vector_y)[0]

# def get_bias(self, A, y):
#     # rcond to avoid numerical errors
#     return np.linalg.lstsq(A, y, rcond=1e-3)

print("\n")

if rerun_bias == 'y':

    # construct the matrix A
    A = np.zeros((training_size, max_user + max_movie)) # M x N matrix
    r = np.zeros(training_size) # 1 x N vector with r
    i = 0
    
    for row in training_indices:
    # for row in training_matrix:
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
        pickle.dump(bias, open( "task{task_no}-bias.p".format(task_no=task_no), "wb"))

else:
    bias = pickle.load(open("task{task_no}-bias.p".format(task_no=task_no), "rb"))

# ---

# # self.baseline_matrix = self.get_baseline_matrix()

# def get_baseline_matrix(self):
#     r_baseline = np.zeros((self.users, self.movies))
#     for user in range(self.users):
#         for movie in range(self.movies):
#             r_sum = self.r_avg + self.bias[user] + self.bias[movie + self.users]
#             # crop values
#             if self.mode == Mode.BASELINE:
#                 if r_sum < 1.0:
#                     r_sum = 1.0
#                 if r_sum > 5.0:
#                     r_sum = 5.0
#             r_baseline[user, movie] = r_sum
#     # round to the nearest integer
#     return r_baseline

training_matrix_predicted = np.zeros((max_user, max_movie))
# for u_index in range(max_user):
#     for m_index in range(max_movie):
#         r_um_predicted = r_avg + bias[u_index] + bias[m_index + max_user]
#         # crop values
#         # if self.mode == Mode.BASELINE: # TODO - MIGHT BE IMPORTANT
#         # if r_um_predicted < 1.0:
#         #     r_um_predicted = 1.0
#         # if r_um_predicted > 5.0:
#         #     r_um_predicted = 5.0
#         training_matrix_predicted[u_index, m_index] = r_um_predicted
for (u_index,m_index), _ in np.ndenumerate(training_matrix_predicted):
    r_um_predicted = r_avg + bias[u_index] + bias[m_index + max_user]
    training_matrix_predicted[u_index, m_index] = r_um_predicted
training_matrix_predicted = np.clip(training_matrix_predicted, 1, 5)
# baseline_matrix is really the predicted matrix for training_matrix

print("\n")
print(f"{training_matrix_predicted=}")
print(f"{r_avg=}")

# ---

# # self.rmse_training = self.get_rmse_training(training_data)

# def get_rmse_training(self, training_set):
#     training_sum = 0.0
#     for user in range(self.users):
#         for movie in range(self.movies):
#             if training_set[user, movie] != 0:
#                 training_sum += (
#                     np.rint(self.baseline_matrix[user, movie])
#                     - training_set[user, movie]
#                 ) ** 2
#     training = m.sqrt(1.0 / self.ratings_number * training_sum)
#     return np.around(training, decimals=3)

# training_set = training_data = training_matrix
# self.baseline_matrix = baseline_matrix
# training_set[user, movie] = r_um_predicted
# self.ratings_number = len(ratings)

# np.rint(self.baseline_matrix[user, movie]) = r_um
# Init: rmse_training = 0
# Loop:
#     mse = (r_um - r_um_predicted) ** 2
# Finally: rmse_training += mse; rmse_training = sqrt(rmse_training / len(ratings))

# =================================================================================
# rmse_training = 0
# for u_index in range(max_user):
#     for m_index in range(max_movie):
#         if training_matrix[u_index, m_index] != 0:
#             r_um_predicted = training_matrix_predicted[u_index, m_index]
#             r_um = training_matrix[u_index, m_index]
#             rmse_training += (r_um - r_um_predicted) ** 2
#             # rmse_training += (
#             #     np.rint(baseline_matrix[u_index, m_index])
#             #     - training_matrix[u_index, m_index]
#             # ) ** 2 # TODO - np.rint is actually not allowed

# rmse_training = sqrt(rmse_training / training_size)
# rmse_training = np.around(rmse_training, decimals=3)
# print("\n")
# print(f"{rmse_training=}")
rmse_training = 0
u_indexes, m_indexes = np.nonzero(training_matrix)
for u_index, m_index in zip(u_indexes, m_indexes):
    r_um_predicted = training_matrix_predicted[u_index, m_index]
    r_um = training_matrix[u_index, m_index]
    rmse_training += (r_um - r_um_predicted) ** 2
rmse_training = sqrt(rmse_training / training_size)
rmse_training = round(rmse_training,3)
print(f"{rmse_training=}")
# =================================================================================

# ---

# # self.rmse_test = self.get_rmse_test(test_data, self.baseline_matrix)

# def get_rmse_test(self, test_set, source):
#     test_sum = 0.0
#     for rating in test_set:
#         test_sum += (np.rint(source[rating[0] - 1, rating[1] - 1]) - rating[2]) ** 2
#     test = m.sqrt(1.0 / len(test_set) * test_sum)
#     return np.around(test, decimals=3)

# =================================================================================
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
# =================================================================================

# ---
# def calculate_absolute_errors(self, test_set, source):
#     filename = "abs_errors_" + self.mode.value + ".png"
#     # plot a histogram
#     hist_data = [
#         (abs(test_set[i][2] - source[test_set[i][0] - 1, test_set[i][1] - 1]))
#         for i in range(len(test_set))
#     ]
#     hist, bins = np.histogram(hist_data, bins=range(10))
#     center = (bins[:-1] + bins[1:]) / 2
#     plt.bar(center, hist, align="center", width=0.7)
#     plt.xlabel("Absolute error")
#     plt.ylabel("Count")
#     plt.title(
#         "Histogram of the distribution of the absolute errors for "
#         + self.mode.value
#         + " predictor\n"
#     )
#     plt.grid(True)
#     plt.savefig(filename)
#     return [x for x in hist if x > 0]

hist, bins = np.histogram(histogram_data, bins=[0,1,2,3,4,5])
plt.figure(0)
plt.bar([0.5, 1.5, 2.5, 3.5, 4.5], hist)
plt.title("Baseline Absolute Errors")
plt.xlabel("Absolute error")
plt.ylabel("Count")
plt.savefig("task{task_no}-baseline_abs_error.png".format(task_no=task_no))
plt.close()
error_dist_baseline = [x for x in hist if x > 0]
print(f"{error_dist_baseline=}")

# exit()

#  ---

# self.difference_matrix = self.get_difference_matrix(training_data)

# def get_difference_matrix(self, training_data):
#     # fill with special values
#     diff_matrix = np.full((self.users, self.movies), NONE)
#     # calculate the difference for each cell
#     for user in range(self.users):
#         for movie in range(self.movies):
#             # if user rated the movie, calculate the difference between actual and predicted grade
#             if training_data[user, movie] != 0.0:
#                 diff_matrix[user, movie] = (
#                     training_data[user, movie] - self.baseline_matrix[user, movie]
#                 )
#     return diff_matrix

# fill with special values
NONE = np.nan
difference_matrix = np.full((max_user, max_user), NONE)
# calculate the difference for each cell
for user in range(max_user):
    for movie in range(max_movie):
        # if user rated the movie, calculate the difference between actual and predicted grade
        if training_matrix[user, movie] != 0.0:
            difference_matrix[user, movie] = (
                training_matrix[user, movie] - training_matrix_predicted[user, movie]
            )

#  ---

# self.distance_matrix = self.calculate_distance_matrix()

# def calculate_distance_matrix(self):
#     # fill with zeroes
#     distance_matrix = np.zeros((self.movies, self.movies))
#     for movie in range(self.movies):
#         distance_matrix[movie, movie] = 0.0
#         # iterate over top triangle of the matrix
#         for candidate in range(movie + 1, self.movies):
#             movie_a = []
#             movie_b = []
#             for user in range(self.users):
#                 # append movies if user rated both of them
#                 if (
#                     self.difference_matrix[user, movie] != NONE
#                     and self.difference_matrix[user, candidate] != NONE
#                 ):
#                     movie_a.append(self.difference_matrix[user, movie])
#                     movie_b.append(self.difference_matrix[user, candidate])
#             # calculate cosine coefficient distance or 0
#             distance_matrix[movie, candidate] = (
#                 1.0 - distance.cosine(movie_a, movie_b)
#                 if len(movie_a) * len(movie_b) > 0
#                 else 0.0
#             )
#             # get bottom triangle by symmetry
#             distance_matrix[candidate, movie] = distance_matrix[movie, candidate]
#     return distance_matrix

if rerun_distance_matrix == 'y':
    distance_matrix = np.zeros((max_movie, max_movie))
    for movie in range(max_movie):
        print('Creating distance_matrix for {movie}/{max_movie}...'.format(
            movie=movie+1, max_movie=max_movie
        ))
        distance_matrix[movie, movie] = 0.0
        # iterate over top triangle of the matrix
        for candidate in range(movie + 1, max_movie):
            movie_a = []
            movie_b = []
            for user in range(max_user):
                # append movies if user rated both of them
                if (
                    difference_matrix[user, movie] != NONE
                    and difference_matrix[user, candidate] != NONE
                ):
                    movie_a.append(difference_matrix[user, movie])
                    movie_b.append(difference_matrix[user, candidate])
            # calculate cosine coefficient distance or 0
            distance_matrix[movie, candidate] = (
                # 1.0 - distance.cosine(movie_a, movie_b)
                distance.cosine(movie_a, movie_b)
                if len(movie_a) > U_MIN
                else 0.0
            )
            # get bottom triangle by symmetry
            distance_matrix[candidate, movie] = distance_matrix[movie, candidate]
    pickle.dump(distance_matrix, open("task{task_no}-distance_matrix-U_MIN-{U_MIN}.p".format(U_MIN=str(U_MIN),task_no=task_no), "wb"))
else:
    distance_matrix = pickle.load(open("task{task_no}-distance_matrix-U_MIN-{U_MIN}.p".format(U_MIN=str(U_MIN),task_no=task_no), "rb"))

#  ---

# self.improved_matrix = self.get_improved_matrix(training_data)

# def get_improved_matrix(self, training_data):
#     improved_matrix = np.zeros((self.users, self.movies))
#     sim = 0.0
#     for movie in range(self.movies):
#         # choose 2 closest neighbours, based on distance matrix
#         neighbours = self.find_best_neighbours(movie, NEIGHBOURS_NUMBER)
#         for user in range(self.users):
#             similarity = self.get_similarity(training_data, user, movie, neighbours)
#             temp_val = self.baseline_matrix[user, movie] + similarity / 10.0
#             sim += similarity
#             # crop values
#             if temp_val < 1.0:
#                 temp_val = 1.0
#             elif temp_val > 5.0:
#                 temp_val = 5.0
#             improved_matrix[user, movie] = temp_val
#     return improved_matrix

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
    else:
        return 0.0
        # return np.nan # TODO potential

def trial_and_error(NEIGHBOURS_NUMBER, PROPORTION_OF_PREDICT, PROPORTION_OF_SIM):
    # NEIGHBOURS_NUMBER = 10
    # SIM_DENOM = 1 # 10.0
    # improved_matrix = np.zeros((max_user, max_movie))
    improved_matrix = np.full([max_user, max_movie], np.nan) # TODO
    sim_data = []
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
    plt.savefig("task{task_no}-improved_abs_error.png".format(task_no=task_no))
    plt.close()
    error_dist_improved = [x for x in hist if x > 0]
    print(f"{error_dist_improved=}")    

# for a,b in itertools.product(a, b):
for NEIGHBOURS_NUMBER, PROPORTION_OF_PREDICT_VS_SIM in itertools.product(NEIGHBOURS_NUMBER_LIST, PROPORTION_OF_PREDICT_VS_SIM_LIST):
    PROPORTION_OF_PREDICT, PROPORTION_OF_SIM = PROPORTION_OF_PREDICT_VS_SIM
    print(f"{U_MIN=}; {NEIGHBOURS_NUMBER=}; {PROPORTION_OF_PREDICT_VS_SIM=}; ", end="")
    trial_and_error(NEIGHBOURS_NUMBER, PROPORTION_OF_PREDICT, PROPORTION_OF_SIM)