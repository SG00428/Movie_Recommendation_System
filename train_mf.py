import pandas as pd
from surprise import SVD, Dataset, Reader
import pickle

ratings = pd.read_csv("data/Netflix_Dataset_Rating.csv")
ratings = ratings.sample(frac=0.1, random_state=42)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['User_ID', 'Movie_ID', 'Rating']], reader)
trainset = data.build_full_trainset()

model = SVD()
model.fit(trainset)

with open("models/svd_model.pkl", "wb") as f:
    pickle.dump(model, f)