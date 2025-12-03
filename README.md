# Movie Recommender System (MovieLens 100k)

This project implements a simple movie recommendation system using: **Python**, **Surprise (SVD collaborative filtering)**, **Streamlit**, **MovieLens 100k dataset**, **Manual anti-testset logic (no build_anti_testset)** and **Direct file paths only (no os, no pathlib)**. The system predicts ratings for unseen movies and recommends the top-N movies to a given user.

## Features

- Collaborative filtering using SVD
- Manual creation of anti-testset
- Streamlit UI for easy interaction
- Clean training and prediction pipeline
- Notebook included for reproducibility

## Dataset Setup

Download MovieLens 100k : https://grouplens.org/datasets/movielens/100k/

Required files:
- u.data
- u.item

## How Recommendations Are Generated

Unlike common Surprise examples, this project does not use build_anti_testset. Instead, this app:
- Retrieves all movie IDs from the training set
- Finds movies already rated by the user
- Creates a custom anti-testset containing only unrated items
- Runs model.test() to generate predictions
- Sorts predictions to produce top-N recommendations
- This approach avoids helper shortcuts and keeps logic explicit.

## Notebook

The notebook movie_recommender_notebook.ipynb provides:
- Data loading
- Model training
- Evaluation (RMSE, MAE)
- Manual anti-testset creation
- Saving model artifacts
