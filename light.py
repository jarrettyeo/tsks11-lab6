import numpy as np
import scipy.sparse as sp

from lightfm.data import Dataset

training_filename = "training/jarettye.training"
test_filename = "test/jarettye.test"
moviename_filename = "moviename/jarettye.moviename"

edge_list = np.loadtxt(training_filename, delimiter=',')
users, movies, ratings = edge_list[:,0], edge_list[:,1], edge_list[:,2]
num_users = int(max(users))
num_items = int(max(movies))
mat = sp.lil_matrix((num_users, num_items), dtype=np.int32)
for u, m, r in zip(users, movies, ratings):
    mat[u-1, m-1] = r
train = mat.tocoo()

edge_list = np.loadtxt(test_filename, delimiter=',')
users, movies, ratings = edge_list[:,0], edge_list[:,1], edge_list[:,2]
mat = sp.lil_matrix((num_users, num_items), dtype=np.int32)
for u, m, r in zip(users, movies, ratings):
    mat[u-1, m-1] = r
test = mat.tocoo()

print(f"{train=}")
print(f"{test=}")

# id_features=<1682x1682 sparse matrix of type '<class 'numpy.float32'>'
#         with 1682 stored elements in Compressed Sparse Row format>
# id_feature_labels=array(['Toy Story (1995)', 'GoldenEye (1995)', 'Four Rooms (1995)', ...,
#        'Sliding Doors (1998)', 'You So Crazy (1994)',
#        'Scream of Stone (Schrei aus Stein) (1991)'], dtype=object)
# genre_features.tocsr()=<1682x19 sparse matrix of type '<class 'numpy.float32'>'
#         with 2893 stored elements in Compressed Sparse Row format>
# genre_feature_labels=array(['genre:unknown', 'genre:Action', 'genre:Adventure',
#        'genre:Animation', "genre:Children's", 'genre:Comedy',
#        'genre:Crime', 'genre:Documentary', 'genre:Drama', 'genre:Fantasy',
#        'genre:Film-Noir', 'genre:Horror', 'genre:Musical',
#        'genre:Mystery', 'genre:Romance', 'genre:Sci-Fi', 'genre:Thriller',
#        'genre:War', 'genre:Western'], dtype='<U17')

