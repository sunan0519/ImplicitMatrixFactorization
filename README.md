# ImplicitMatrixFactorization


## Data
Raw data from https://labrosa.ee.columbia.edu/millionsong/tasteprofile.

It is composed of more than 48 million triplets (user, song, listening frequency) recovered from the listening user histories. Data were provided from several applications, where each user can listen to the music he wanted. Data from 1.2 million users listening to 380,000 songs.

This dataset is considered as implicit feedback dataset.

## Methods
This project focused on the collaborative filtering approach. The method and evaluation metric of implicit matrix factorization implemented in this project is outlined in [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf). 

## Implement
The python file `proprocessing.py`
transforms raw data to a user-song sparse matrix and takes a subset given threshold of users and songs(the number of songs a user should has listened to and the number of times a song has ever been listened to). The result sparse matrix is stored in a npz file: `sample_matrix.npz`
The Implicit Matrix Factorization method is given in: `implicit_mf.py`

Parameter tuning is finished in :`main.py`

## Update
The model performance is optimized now in `MF-final_version`. Cython, Conjugate Gradient method(instead of ALS) and Python performance tips like reducing function call overhead are used. It is at least 5 times faster than original version. (27 seconds to finish one iteration now and 180s originally).
