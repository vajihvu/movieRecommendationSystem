import pandas as pd
from collections import defaultdict
from surprise import Dataset, Reader
import pickle

def load_movielens():
    try:
        data = Dataset.load_builtin("ml-100k")
    except:
        df = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["user_id","item_id","rating","timestamp"])
        reader = Reader(rating_scale=(1,5))
        data = Dataset.load_from_df(df[["user_id","item_id","rating"]], reader)
    return data

def load_movie_titles():
    try:
        df = pd.read_csv("data/ml-100k/u.item", sep="|", header=None, encoding="latin-1")
        df = df[[0,1]]
        df.columns = ["item_id","title"]
        return dict(zip(df.item_id, df.title))
    except:
        return {}

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid in top_n:
        top_n[uid].sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = top_n[uid][:n]
    return top_n