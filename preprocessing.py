import numpy as np
import scipy.sparse as sp

def unique_songs(file):
    """ This function returns a set of unique user
    - param:
          file : training file
    """
    s = set()
    with open(file, "r") as f:
        for line in f:
            _, song, _ = line.strip().split('\t')
            if song not in s:
                s.add(song)
    return s


def unique_users(file):
    """ This function returns a set of unique user
    - param:
          file : training file
    """
    u = set()
    with open(file, "r") as f:
        for line in f:
            user, _, _ = line.strip().split('\t')
            if user not in u:
                u.add(user)
    return u
    
#Load unique Users indexes
def uid_to_index(file):
    uniq_users = unique_users(file)
    #Enumerate User with indexes
    u_i = dict()
    for i, u in enumerate(uniq_users):
        u_i[u] = i
    return u_i
    
def sid_to_index(file):
    #Load unique Songs indexes
    uniq_songs = unique_songs(file)
    #Enumerate Songs with indexes
    s_i = dict()
    for i, s in enumerate(uniq_songs):
        s_i[s] = i
    return s_i
    
def load_data(file):
    u_i = uid_to_index(file)
    s_i = sid_to_index(file)
    dok_mat = sp.dok_matrix((1019318,384546), dtype=np.int8)
    with open(file, "r") as f:
        for line in f:
            user, song, count = line.strip().split('\t')
            dok_mat[u_i[user], s_i[song]] = count
    
    sparse_mat = dok_mat.tocsr()
    return sparse_mat

def prep(user_song_matrix, user_thres = 200, song_thres = 200):
    col_index_matrix = user_song_matrix.copy()
    col_index_matrix[col_index_matrix.nonzero()] = 1
    
    col_sum = np.squeeze(np.asarray(col_index_matrix.sum(axis=0)))
    col_mask = col_sum > song_thres 
    delete_song = user_song_matrix[:, col_mask]
    
    row_index_matrix = delete_song.copy()
    row_index_matrix[row_index_matrix.nonzero()] = 1
    
    row_sum = np.squeeze(np.asarray(row_index_matrix.sum(axis=1)))
    row_mask = row_sum > user_thres  
    
    delete_user_song = delete_song[row_mask, :]
#    user_song_normalized = lil_matrix(normalize(delete_user_song, axis=1))
#    return user_song_normalized
    return delete_user_song
    
sparse_mat = load_data('train_triplets.txt')
sp.save_npz('sparse_matrix.npz', sparse_mat)
sample_matrix = prep(sparse_mat, user_thres = 200, song_thres = 1200)
sp.save_npz('sample_matrix.npz', sample_matrix)
