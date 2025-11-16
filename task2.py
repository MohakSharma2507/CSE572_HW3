import pandas as pd
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import cross_validate

# --------------------------------------------------
# Load Data
# --------------------------------------------------

df = pd.read_csv("ratings_small.csv")

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

# --------------------------------------------------
# Models
# --------------------------------------------------

def evaluate_model(model, name):
    print(f"\nRunning {name}")
    scores = cross_validate(model, data, measures=["MAE", "RMSE"], cv=5, verbose=True)
    return scores

# User-based CF
user_cf = KNNBasic(sim_options={"name": "cosine", "user_based": True})

# Item-based CF
item_cf = KNNBasic(sim_options={"name": "cosine", "user_based": False})

# Probabilistic Matrix Factorization
pmf = SVD()

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

if __name__ == "__main__":
    scores_user = evaluate_model(user_cf, "User-Based CF")
    scores_item = evaluate_model(item_cf, "Item-Based CF")
    scores_pmf = evaluate_model(pmf, "PMF")