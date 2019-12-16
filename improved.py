import numpy as np
from numpy import dot
from numpy.linalg import norm

from scipy.spatial.distance import cosine

from matplotlib import pyplot as plt 
from math import sqrt

from yaspin import yaspin
import pickle

import sys

rerun_bias = input("Rerun bias? (y/n) ")
assert rerun_bias in ['y','n']

rerun_cosine_distance_matrix = input("Rerun cosine_distance_matrix? (y/n) ")
assert rerun_bias in ['y','n']

number_of_closest_neighbours = 2

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

with yaspin(text = "Creating Training Matrix Predicted...").white.bold.shark.on_blue as sp:
    training_matrix_predicted = np.zeros((max_user, max_movie))
    for (u_index,m_index), _ in np.ndenumerate(training_matrix_predicted):
        r_um_predicted = r_avg + bias[u_index] + bias[m_index + max_user]
        training_matrix_predicted[u_index, m_index] = r_um_predicted
    training_matrix_predicted = np.clip(training_matrix_predicted, 1, 5)
    # baseline_matrix is really the predicted matrix for training_matrix

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

hist, bins = np.histogram(histogram_data, bins=[0,1,2,3,4,5])
plt.bar([0.5, 1.5, 2.5, 3.5, 4.5], hist)
plt.title("Improved Absolute Errors")
plt.xlabel("Absolute error")
plt.ylabel("Count")
plt.savefig("improved_abs_error.png")
error_dist = [x for x in hist if x > 0]
print(f"{error_dist=}")

# ------------------------------------------------------------------------------- #
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
with yaspin(text = "Creating Difference Matrix...").white.bold.shark.on_blue as sp:
    difference_matrix = np.full((max_user, max_movie), np.nan)
    u_indexes, m_indexes = np.nonzero(training_matrix)
    for u_index, m_index in zip(u_indexes, m_indexes):
        difference_matrix[u_index, m_index] = training_matrix[u_index, m_index] - training_matrix_predicted[u_index, m_index]
    number_of_nan_values = np.count_nonzero(np.isnan(difference_matrix))
    number_of_non_nan_values = np.count_nonzero(~np.isnan(difference_matrix))
    print(f"{difference_matrix=}")
    print(f"{number_of_nan_values=}")
    print(f"{number_of_non_nan_values=}")
# ------------------------------------------------------------------------------- #
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

# ---
# with yaspin(text = "Creating Cosine Distance Matrix...").white.bold.shark.on_blue as sp:
if rerun_cosine_distance_matrix == 'y':
    cosine_distance_matrix = np.zeros((max_movie, max_movie))
    # for m1, m2 in [(m1, m2) for m1 in range(max_movie) for m2 in range(m1+1, max_movie) if m1 != m2]:
    for m1 in range(max_movie):
        print('Creating Cosine Distance Matrix for {m1}/{max_movie}...'.format(
            m1=m1+1, max_movie=max_movie
        ))
        for m2 in range(m1+1, max_movie):
            if m1 != m2:
                # print('Creating Cosine Distance Matrix... {m1}/{max_movie} out of {m2}/{max_movie}'.format(
                #     m1=m1, m2=m2, max_movie=max_movie
                # ))
                # append movies if user rated both of them
                try:
                    m1_indices, m2_indices = zip(*[(difference_matrix[u,m1], difference_matrix[u,m2]) for u in range(max_user) if not np.isnan(difference_matrix[u,m1]) and not np.isnan(difference_matrix[u,m2])])
                except ValueError as e:
                    # if 'nan' in str(e):
                    #     print(m1, m2)
                    #     print(str(e))
                    #     # x = input('Enter to continue... ')
                    continue
                # m1_indices = [difference_matrix[u,m1] for u in range(max_user) if not np.isnan(difference_matrix[u,m1]) and not np.isnan(difference_matrix[u,m2])]
                # m2_indices = [difference_matrix[u,m2] for u in range(max_user) if not np.isnan(difference_matrix[u,m1]) and not np.isnan(difference_matrix[u,m2])]

                # calculate cosine distance
                cosine_distance_matrix[m1, m2] = cosine_distance_matrix[m2, m1] = dot(m1_indices, m2_indices) / (norm(m1_indices) * norm(m2_indices))
                # cosine_distance_matrix[m1, m2] = (
                #         1.0 - cosine(m1_indices, m2_indices)
                #         if len(m1_indices) * len(m2_indices) > 0
                #         else 0.0
                #     )
                # cosine_distance_matrix[m2, m1] = cosine_distance_matrix[m1, m2]
    pickle.dump(cosine_distance_matrix, open("cosine_distance_matrix.p", "wb"))
else:
    cosine_distance_matrix = pickle.load(open("cosine_distance_matrix.p", "rb"))
print(f"{cosine_distance_matrix=}")
# ---
# # with yaspin(text = "Creating Cosine Distance Matrix...").white.bold.shark.on_blue as sp:
# if rerun_cosine_distance_matrix == 'y':
#     cosine_distance_matrix = np.zeros((max_movie, max_movie))
#     m1_indices, m2_indices = np.isfinite(difference_matrix)
#     # for m1, m2 in [(m1, m2) for m1 in range(max_movie) for m2 in range(m1+1, max_movie) if m1 != m2]:
#     for m1 in range(max_movie):
#         print('Creating Cosine Distance Matrix for {m1}/{max_movie}...'.format(
#             m1=m1, max_movie=max_movie
#         ))        
#         for m2 in range(m1+1, max_movie):
#             if m1 != m2:
#                 # print('Creating Cosine Distance Matrix... {m1}/{max_movie} out of {m2}/{max_movie}'.format(
#                 #     m1=m1, m2=m2, max_movie=max_movie
#                 # ))
#                 print("\033c", end="")
#                 # append movies if user rated both of them
#                 try:
#                     m1_indices, m2_indices = zip(*[(difference_matrix[u,m1], difference_matrix[u,m2]) for u in range(max_user) if not np.isnan(difference_matrix[u,m1]) and not np.isnan(difference_matrix[u,m2])])
#                 except ValueError as e:
#                     print(m1, m2)
#                     print(str(e))
#                     continue
#                 # m1_indices = [difference_matrix[u,m1] for u in range(max_user) if not np.isnan(difference_matrix[u,m1]) and not np.isnan(difference_matrix[u,m2])]
#                 # m2_indices = [difference_matrix[u,m2] for u in range(max_user) if not np.isnan(difference_matrix[u,m1]) and not np.isnan(difference_matrix[u,m2])]

#                 # calculate cosine distance
#                 cosine_distance_matrix[m1, m2] = cosine_distance_matrix[m2, m1] = dot(m1_indices, m2_indices) / (norm(m1_indices) * norm(m2_indices)) if len(m1_indices) > 0 else np.nan
#                 # cosine_distance_matrix[m1, m2] = (
#                 #         1.0 - cosine(m1_indices, m2_indices)
#                 #         if len(m1_indices) * len(m2_indices) > 0
#                 #         else 0.0
#                 #     )
#                 # cosine_distance_matrix[m2, m1] = cosine_distance_matrix[m1, m2]
#     pickle.dump(cosine_distance_matrix, open( "cosine_distance_matrix.p", "wb"))
# else:
#     cosine_distance_matrix = pickle.load(open("cosine_distance_matrix.p", "rb"))
# print(f"{cosine_distance_matrix=}")
# ------------------------------------------------------------------------------- #
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
with yaspin(text = "Calculating similarities...").white.bold.shark.on_blue as sp:
    improved_matrix_predicted = np.zeros((max_user, max_movie))
    for u, m in zip(range(max_movie), range(max_user)):
        # choose 2 closest neighbours, based on distance matrix
        neighbours = np.argsort([abs(x) for x in cosine_distance_matrix[m]])[::-1][:number_of_closest_neighbours]
        sim_denominator = 0.0
        sim_numerator = 0.0
        similarity = 0.0
        for neighbour in neighbours:
            if training_matrix[u, neighbour] != 0:
                sim_numerator += (
                    cosine_distance_matrix[m, neighbour]
                    * difference_matrix[u, neighbour]
                )
                sim_denominator += abs(cosine_distance_matrix[m, neighbour])
        if sim_denominator != 0:
            similarity += sim_numerator * 1.0 / sim_denominator
        temp_val = training_matrix_predicted[u, m] + similarity / 10.0
        improved_matrix_predicted[u, m] = temp_val
    improved_matrix_predicted = np.clip(improved_matrix_predicted, 1, 5)
    print(f"{improved_matrix_predicted=}")
# ------------------------------------------------------------------------------- #
# self.rmse_test_improved = self.get_rmse_test(test_data, self.improved_matrix)
# def get_rmse_test(self, test_set, source):
#     test_sum = 0.0
#     for rating in test_set:
#         test_sum += (np.rint(source[rating[0] - 1, rating[1] - 1]) - rating[2]) ** 2
#     test = m.sqrt(1.0 / len(test_set) * test_sum)
#     return np.around(test, decimals=3)
rmse_training = 0
u_indexes, m_indexes = np.nonzero(training_matrix)
for u_index, m_index in zip(u_indexes, m_indexes):
    r_um_predicted = improved_matrix_predicted[u_index, m_index]
    r_um = training_matrix[u_index, m_index]
    rmse_training += (r_um - r_um_predicted) ** 2
rmse_training = sqrt(rmse_training / training_size)
rmse_training = round(rmse_training,3)
print(f"{rmse_training=}")
# ------------------------------------------------------------------------------- #
rmse_test = 0
histogram_data = []
u_indexes, m_indexes = np.nonzero(test_matrix)
for u_index, m_index in zip(u_indexes, m_indexes):
    r_um_predicted = improved_matrix_predicted[u_index, m_index]
    r_um = test_matrix[u_index, m_index]
    abs_error = abs(r_um - int(r_um_predicted))
    histogram_data.append(abs_error)
    rmse_test += (r_um - r_um_predicted) ** 2
rmse_test = sqrt(rmse_test / test_size)
rmse_test = round(rmse_test,3)
print(f"{rmse_test=}")
# ------------------------------------------------------------------------------- #
hist, bins = np.histogram(histogram_data, bins=[0,1,2,3,4,5])
plt.bar([0.5, 1.5, 2.5, 3.5, 4.5], hist)
plt.title("Improved Absolute Errors")
plt.xlabel("Absolute error")
plt.ylabel("Count")
plt.savefig("improved_abs_error.png")
error_dist = [x for x in hist if x > 0]
print(f"{error_dist=}")