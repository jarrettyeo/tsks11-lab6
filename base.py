import numpy as np
from matplotlib import pyplot as plt 

from math import sqrt

from yaspin import yaspin
import pickle

rerun_bias = input("Rerun bias? (y/n) ")
assert rerun_bias in ['y','n']

training_filename = "training/jarettye.training"
test_filename = "test/jarettye.test"
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
        pickle.dump(bias, open( "bias.p", "wb"))

else:
    bias = pickle.load(open("bias.p", "rb"))

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
plt.bar([0.5, 1.5, 2.5, 3.5, 4.5], hist)
plt.title("Baseline Absolute Errors")
plt.xlabel("Absolute error")
plt.ylabel("Count")
plt.savefig("baseline_abs_error.png")
error_dist = [x for x in hist if x > 0]
print(f"{error_dist=}")

#  ---