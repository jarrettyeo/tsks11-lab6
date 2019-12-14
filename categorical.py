# 1. Form dict d of movies -> list of categories
# 2. For each user, for each category, take average of all ratings user has rated for that category

import numpy as np
from matplotlib import pyplot as plt 

from math import sqrt

from yaspin import yaspin
import pickle
import csv

rerun_bias = input("Rerun bias? (y/n) ")
assert rerun_bias in ['y','n']

rerun_bias_c = input("Rerun bias_c? (y/n) ")
assert rerun_bias_c in ['y','n']

training_filename = "training/jarettye.training"
test_filename = "test/jarettye.test"
moviename_filename = "moviename/jarettye.moviename"
category_bias_proportion = 0.1

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

def generate_matrix_movienames(filename, training_matrix, max_user, max_movie):
    print("\n")
    c_reader = csv.reader(open(filename, 'r'), delimiter=';')
    columns = list(zip(*c_reader))

    movie_indexes, movie_names, movie_categories = columns[0], columns[1], columns[2]
    movie_indexes = [movie_index.strip() for movie_index in movie_indexes]
    movie_indexes = [int(movie_index) for movie_index in movie_indexes]
    movie_names = [movie_name.strip() for movie_name in movie_names]
    movie_categories = [movie_category.strip() for movie_category in movie_categories]

    movie_id_to_category_name_list = {}

    for i in range(len(movie_indexes)):
        movie_id_to_category_name_list[int(movie_indexes[i])] = movie_categories[i].split('|')
    # print(f"{movie_id_to_category_name_list=}")

    category_name_set = set()
    for category_list in movie_id_to_category_name_list.values():
        for category_name in category_list:
            category_name_set.add(category_name)
    print(f"{category_name_set=}")
    
    max_category = len(category_name_set)

    category_id_to_category_name = {}
    category_name_list = list(category_name_set)
    for i in range(len(category_name_list)):
        category_id_to_category_name[i] = category_name_list[i]
    # print(f"{category_id_to_category_name=}")
    
    movie_id_to_category_id_list = {}
    for movie_index, this_category_name_list in movie_id_to_category_name_list.items():
        movie_id_to_category_id_list[movie_index] = []
        for this_category_name in this_category_name_list:
            for category_id, category_name in category_id_to_category_name.items():
                if category_name == this_category_name:
                    movie_id_to_category_id_list[movie_index].append(category_id)
                    break
    
    with yaspin(text = "Mapping user ratings to movie categories...").white.bold.shark.on_blue as sp:
        user_category_rating_sum_matrix = np.zeros((max_user, max_category))
        user_category_rating_count_matrix = np.zeros((max_user, max_category))
        u_and_m_indices = np.transpose(np.nonzero(training_matrix)) 
        for u_and_m_indexes_pair in u_and_m_indices: # u_index and m_index are [0,N-1]
            u_index = u_and_m_indexes_pair[0]
            m_index = u_and_m_indexes_pair[1]
            rating = training_matrix[u_index][m_index]
            category_id_list = movie_id_to_category_id_list[m_index+1]
            # print(f"{u_index=}")
            # print(f"{m_index=}")
            # print(f"{rating=}")
            # print(f"{category_id_list=}")
            # x = input("Press enter to continue... ")
            for c_index in category_id_list:
                user_category_rating_sum_matrix[u_index][c_index] += rating
                user_category_rating_count_matrix[u_index][c_index] += 1
        user_category_rating_avg_matrix = np.nan_to_num(user_category_rating_sum_matrix / user_category_rating_count_matrix) # convert NaN to 0
    print('\n')
    return user_category_rating_avg_matrix, max_category, movie_id_to_category_id_list

training_matrix, training_size, training_max_user, training_max_movie = generate_matrix(training_filename)
test_matrix, test_size, test_max_user, test_max_movie = generate_matrix(test_filename)

max_user = max(training_max_user, test_max_user)
max_movie = max(training_max_movie, test_max_movie)

user_category_rating_avg_matrix, max_category, movie_id_to_category_id_list = generate_matrix_movienames(moviename_filename, training_matrix, max_user, max_movie)

print(f"{training_size=}")
print(f"{test_size=}")
print(f"{max_user=}")
print(f"{max_movie=}")
print(f"{max_category=}")
print(f"{user_category_rating_avg_matrix=}")

training_indices = np.nonzero(training_matrix) # find the indices of an array, where a condition is True
training_indices = np.transpose(training_indices)

r_sum = np.sum(training_matrix)
r_avg = r_sum / training_size

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
        pickle.dump(bias, open( "bias.p", "wb"))

else:
    bias = pickle.load(open("bias.p", "rb"))

training_indices_c = np.nonzero(user_category_rating_avg_matrix) # find the indices of an array, where a condition is True
training_indices_c = np.transpose(user_category_rating_avg_matrix)

user_category_rating_avg_size = np.count_nonzero(user_category_rating_avg_matrix)

r_sum_c = np.sum(user_category_rating_avg_matrix)
r_avg_c = r_sum_c / user_category_rating_avg_size

if rerun_bias_c == 'y':

    A = np.zeros((user_category_rating_avg_size, max_user + max_category)) # M x N matrix
    r = np.zeros(user_category_rating_avg_size) # 1 x N vector with r
    i = 0
    
    for row in training_indices_c:
        A[i, int(row[0])] = 1
        A[i, int(row[1]) + max_user] = 1
        r[i] = user_category_rating_avg_matrix[int(row[0]), int(row[1])] - r_avg_c
        i += 1

    print("\n")
    print(f"{A=}")
    print(f"{r=}")

    with yaspin(text = "Running least-squares solution to linear matrix equation (category) ...").white.bold.shark.on_blue as sp:
        bias_c, _, _, _ = np.linalg.lstsq(A, r, rcond=1.e-3)
        print(f"\n{bias_c=}")
        pickle.dump(bias_c, open( "bias_c.p", "wb"))

else:
    bias_c = pickle.load(open("bias_c.p", "rb"))

training_matrix_predicted = np.zeros((max_user, max_movie))

with yaspin(text = "Calculating r_um_predicted...").white.bold.shark.on_blue as sp:
    for (u_index,m_index), _ in np.ndenumerate(training_matrix_predicted):

        # User-Movie Bias Calculation        
        r_um_predicted = r_avg + bias[u_index] + bias[m_index + max_user]
        r_um_predicted = 1 if r_um_predicted < 1 else 5 if r_um_predicted > 5 else r_um_predicted

        # User-Category Bias Calculation
        c_indexes = movie_id_to_category_id_list[m_index+1]
        sum_bias_c = sum([bias_c[c_index + max_user] for c_index in c_indexes])
        count_bias_c = len(c_indexes)
        avg_bias_c = sum_bias_c / count_bias_c
        r_um_predicted_c = r_avg_c + bias_c[u_index] + avg_bias_c
        r_um_predicted_c = 1 if r_um_predicted_c < 1 else 5 if r_um_predicted_c > 5 else r_um_predicted_c

        training_matrix_predicted[u_index, m_index] = category_bias_proportion * r_um_predicted_c + (1 - category_bias_proportion) * r_um_predicted
    training_matrix_predicted = np.clip(training_matrix_predicted, 1, 5)
    # baseline_matrix is really the predicted matrix for training_matrix

print("\n")
print(f"{training_matrix_predicted=}")
print(f"{r_avg=}")
print(f"{r_avg_c=}")

rmse_training = 0
u_indexes, m_indexes = np.nonzero(training_matrix)
for u_index, m_index in zip(u_indexes, m_indexes):
    r_um_predicted = training_matrix_predicted[u_index, m_index]
    r_um = training_matrix[u_index, m_index]
    error = category_bias_proportion * (r_um - int(r_um_predicted))
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
    error = category_bias_proportion * (r_um - int(r_um_predicted))
    abs_error = abs(error)
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
plt.savefig("baseline_abs_error.png")
error_dist = [x for x in hist if x > 0]
print(f"{error_dist=}")