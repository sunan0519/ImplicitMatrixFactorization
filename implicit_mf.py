import time
from scipy.sparse.linalg import spsolve
import numpy as np
import scipy.sparse as sp


class ImplicitMF():

    def __init__(self, counts, alpha, num_factors=40, num_iterations=30,
                 reg_param=0.8):
        self.counts = counts
        self.alpha = alpha
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param

    def fit(self):
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))

        for i in range(self.num_iterations):
#            t0 = time.time()
#            print ('Solving for user vectors...')
            self.user_vectors = self.iteration(True, sp.csr_matrix(self.item_vectors))
#            print ('Solving for item vectors...')
            self.item_vectors = self.iteration(False, sp.csr_matrix(self.user_vectors))
#            t1 = time.time()
#            print ('iteration %i finished in %f seconds' % (i + 1, t1 - t0))

    def iteration(self, user, fixed_vecs):
        num_solve = self.num_users if user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        eye = sp.eye(num_fixed)
        lambda_eye = self.reg_param * sp.eye(self.num_factors)
        solve_vecs = np.zeros((num_solve, self.num_factors))

#        t = time.time()
        for i in range(num_solve):
            if user:
                counts_i = self.counts[i].toarray()
            else:
                counts_i = self.counts[:, i].T.toarray()
            CuI = sp.diags(1 + self.alpha * np.log(1 + counts_i), [0])
            pu = counts_i.copy()
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sp.csr_matrix(pu).T)
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu
#            if i % 1000 == 0:
#                print ('Solved %i vecs in %d seconds' % (i, time.time() - t))
#                t = time.time()

        return solve_vecs
    
    def predict(self, u, i):
        """ Single user and item prediction. """
        return self.user_vectors[u, :].dot(self.item_vectors[i, :].T)
    
    def predict_all(self):
        """ Predict ratings for every user and item. """
        predictions = np.zeros((self.user_vectors.shape[0],self.item_vectors.shape[0]))
        for u in range(self.user_vectors.shape[0]):
            for i in range(self.item_vectors.shape[0]):
                predictions[u, i] = self.predict(u, i)
        return predictions
    
    def ranking(self, predictions):
        temp = predictions.argsort(axis = 1)
        #produce the abosulte ranks for each item for each user
        pred_ranks = np.empty_like(temp)
        for i in range(self.num_users):
            pred_ranks[i,temp[i,:]] = np.arange(self.num_items - 1, -1, -1)
        #convert the ranks to rank percentile
        pred_ranks_percentile = pred_ranks / np.max(pred_ranks) * 100
        return pred_ranks_percentile
    
    def evaluate(self, test):
        predictions = self.predict_all()
        pred_ranks = self.ranking(predictions)
        test = test.todense()
        metrics = np.sum(np.multiply(test, pred_ranks))/np.sum(test)
        return metrics
    
    
    
    
