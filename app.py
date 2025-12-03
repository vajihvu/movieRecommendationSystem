import pickle
import streamlit as st
import pandas as pd
from utils import load_movielens, load_movie_titles, get_top_n

st.title("Movie Recommendendation System")

model = None
try:
    with open("src/model/svd_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception:
    st.error("Model not found. Run training first.")
    st.stop()

try:
    with open("src/model/movie_titles.pkl", "rb") as f:
        movie_titles = pickle.load(f)
except Exception:
    movie_titles = {}

user_id = st.text_input("Enter user ID (1 - 943):", "1").strip()
num_recs = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Recommend"):
    if not hasattr(model, "trainset") or model.trainset is None:
        st.error("Loaded model does not contain a trainset. Re-train & save the Surprise algorithm object.")
        st.stop()

    trainset = model.trainset

    all_items = [trainset.to_raw_iid(i) for i in trainset.all_items()]

    user_rated_items = set()
    try:
        inner_uid = trainset.to_inner_uid(user_id)
        user_rated_items = set(trainset.to_raw_iid(inner_iid) for (inner_iid, _) in trainset.ur[inner_uid])
    except Exception:
        user_rated_items = set()

    anti = [(user_id, item, 0.0) for item in all_items if item not in user_rated_items]

    preds = model.test(anti)
    top_n = get_top_n(preds, num_recs)

    recs = top_n.get(user_id, [])
    if not recs:
        st.warning("No recommendations found for this user id - it may be out of range of the training set.")
    else:
        rows = []
        for iid, est in recs:
            try:
                title = movie_titles.get(int(iid), movie_titles.get(iid, f"Item {iid}"))
            except Exception:
                title = movie_titles.get(iid, f"Item {iid}")
            rows.append({"Movie": title, "Prediction Rating": round(est, 2)})

        st.table(pd.DataFrame(rows))