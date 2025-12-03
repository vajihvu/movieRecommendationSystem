import pickle
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from utils import load_movielens, load_movie_titles

def main():
    data = load_movielens()
    titles = load_movie_titles()

    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)

    preds = model.test(testset)
    rmse = accuracy.rmse(preds)
    print("RMSE =", rmse)

    with open("src/model/svd_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("src/model/movie_titles.pkl", "wb") as f:
        pickle.dump(titles, f)

if __name__ == "__main__":
    main()