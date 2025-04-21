# models/matrix_factorization.py
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

def train_svd_model(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['User_ID', 'Movie_ID', 'Rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2)
    algo = SVD()
    algo.fit(trainset)
    return algo
