import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix


class NeighborhoodModel(object):

	def __init__(self):
		
		self.pred_ranks_percentile = None
		self.pred_ranks = None
		self.num_instances_train, self.num_features_train = None, None
		#case amplification factor

	def fit(self, train, test, rho):
		self.train = train
		self.test = test
		pred = self.predict(self.train, rho)
		self.pred_ranks, self.pred_ranks_percentile = self.ranking(pred)

	#output the predicted score for each item for each user
	def predict(self, x, rho):
		v_bar = lil_matrix(x.sum(axis = 1))
		weight = (x.dot(x.T)).multiply(v_bar.dot(v_bar.T).power(-1/2)).power(rho)

		for i in range(weight.shape[0]):
			weight[i,i] = 0
		pred = weight.dot(x).todense()
		return pred

	def predict_ind(self, x, user_pref_sparse):
		v_bar = lil_matrix(x.sum(axis = 1))
		weight = (user_pref_sparse.dot(x.T)).multiply(v_bar.power(-1/2))
		for i in range(weight.shape[0]):
			weight[i,i] = 0
		pred = weight.dot(x).todense()
		return pred

	#produce the ranking percentile for each item for each user
	def ranking(self, pred):
		num_instances_train, num_features_train = pred.shape[0], pred.shape[1]
		temp = pred.argsort(axis = 1)
		#produce the abosulte ranks for each item for each user
		pred_ranks = np.empty_like(temp)
		for i in range(num_instances_train):
			pred_ranks[i,temp[i,:]] = np.arange(num_features_train - 1, -1, -1)
		#convert the ranks to rank percentile
		pred_ranks_percentile = pred_ranks / np.max(pred_ranks) * 100
		return pred_ranks, pred_ranks_percentile
	
	#output expected percentile ranking of a watching unit
	def evaluate(self):
		test = self.test

		num_instances_train, num_features_train = self.num_instances_train, self.num_features_train
		pred_ranks_percentile = self.pred_ranks_percentile
		test = test.todense()
		metrics = np.sum(np.multiply(test, pred_ranks_percentile))/np.sum(test)
		return metrics

	#recommend the top "num_rec" songs to user "user_id"
	def recommend(self, data, user_id,rho = 1, num_rec = 3):
		pred = self.predict(data, rho)
		pred_ranks, pred_ranks_percentile = self.ranking(pred)
		song_rank_list = np.asarray(pred_ranks[user_id,:]).squeeze()
		#produce the song list sorted by their scores
		rank_index = np.argsort(song_rank_list)
		rec_list = []
		num = 0
		song_arr = np.asarray(data[user_id,:].todense()).squeeze()
		#songs that the user has already listened
		song_in_bucket = np.nonzero(song_arr)[0]
		for item in rank_index:
			if num >= num_rec:
				break
			#exclude the songs that the user has already listened
			if item not in song_in_bucket:
				rec_list.append(item)
				num += 1

		return rec_list

#recommend songs for a user not in the data
#input a array of the times of the songs that the user has listened
	def recommend_out(self, data, user_pref, num_rec = 3):
		user_pref_sparse = lil_matrix(user_pref, dtype = np.float64)
		#similarity_ind = user_pref_sparse.dot(data.T)
		#pred = similarity_ind.dot(data).todense()
		pred = self.predict_ind(data, user_pref_sparse)[0]
		song_rank_list, _ = np.asarray(self.ranking(pred)).squeeze()
		rank_index = np.argsort(song_rank_list)
		rec_list = []
		num = 0
		song_arr = np.asarray(user_pref).squeeze()
		#songs that the user has already listened
		song_in_bucket = np.nonzero(song_arr)[0]
		for item in rank_index:
			if num >= num_rec:
				break
			#exclude the songs that the user has already listened
			if item not in song_in_bucket:
				rec_list.append(item)
				num += 1
		return rec_list

