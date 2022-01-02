
import gc
import re
import scipy
import numpy as np
import pandas as pd
from scipy import sparse
from pprint import pprint
from lightgbm import LGBMRegressor
from IPython.display import display, HTML
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import copy

import numpy as np
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords

stop = stopwords.words("english")
warnings.filterwarnings("ignore")

pd.options.display.max_colwidth = 300

# to change
# save_dir = "/kaggle/working"
save_dir = "../save"

def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat


# # Training data
# ## Convert the label to SUM of all toxic labels (This might help with maintaining toxicity order of comments)
df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")

# ## Load Validation and Test data
df_val = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data.csv")
df_sub = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")

df["severe_toxic"] = df.severe_toxic * 2
df["y"] = (
    df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(
        axis=1
    )
).astype(int)
df["y"] = df["y"] / df["y"].max()
df = df[["comment_text", "y"]].rename(columns={"comment_text": "text"})
df.sample(5)
df["y"].value_counts()

# ## Create 3 versions of the data
n_folds = 7
ruddit_path = "../input/ruddit-jigsaw-dataset/Dataset"

def clean(data, col):
    data[col] = data[col].str.replace(r"what's", "what is ")
    data[col] = data[col].str.replace(r"\'ve", " have ")
    data[col] = data[col].str.replace(r"can't", "cannot ")
    data[col] = data[col].str.replace(r"n't", " not ")
    data[col] = data[col].str.replace(r"i'm", "i am ")
    data[col] = data[col].str.replace(r"\'re", " are ")
    data[col] = data[col].str.replace(r"\'d", " would ")
    data[col] = data[col].str.replace(r"\'ll", " will ")
    data[col] = data[col].str.replace(r"\'scuse", " excuse ")
    data[col] = data[col].str.replace(r"\'s", " ")
    data[col] = data[col].str.replace("\n", " \n ")
    data[col] = data[col].str.replace(r"([a-zA-Z]+)([/!?.])([a-zA-Z]+)", r"\1 \2 \3")
    data[col] = data[col].str.replace(r"([*!?\'])\1\1{2,}", r"\1\1\1")
    data[col] = data[col].str.replace(r"([*!?\']+)", r" \1 ")
    data[col] = data[col].str.replace(r"([a-zA-Z])\1{2,}\b", r"\1\1")
    data[col] = data[col].str.replace(r"([a-zA-Z])\1\1{2,}\B", r"\1\1\1")
    data[col] = data[col].str.replace(r"[ ]{2,}", " ").str.strip()
    data[col] = data[col].str.replace(r"[ ]{2,}", " ").str.strip()
    data[col] = data[col].apply(
        lambda x: " ".join([word for word in x.split() if word not in (stop)])
    )
    return data

def add_df_sub(df, random_state):
    df_sub_to_add = copy.copy(df_sub)
    df_sub_to_add['y'] = np.nan
    df_sub_to_add = df_sub_to_add[['text', 'y']]
    df_sub_to_add = df_sub_to_add.sample(frac=0.7, random_state=random_state)
    return pd.concat([df, df_sub_to_add], axis=0)


def get_nan_index_list(df):
    df = df.reset_index(drop=True)
    nan_index_list = list(df[df.y==np.nan].index)
    if len(nan_index_list) == 0 and len(list(df[df.y.map(np.isnan)]))>0:
        nan_index_list = list(df[df.y.map(np.isnan)].index)
    df.dropna(inplace=True)
    return df, nan_index_list

def save_raw_df(df):
    frac_1 = 0.3
    frac_1_factor = 1.2

    for fld in range(n_folds):
        print(f"Fold: {fld}")
        df_sampled1 = df[df.y > 0].sample(frac=frac_1, random_state=10 * (fld + 1))
        df_sampled2 = df[df.y == 0].sample(
            n=int(len(df[df.y > 0]) * frac_1 * frac_1_factor),
            random_state=10 * (fld + 1),
        )
        df_concat = pd.concat([df_sampled1, df_sampled2], axis=0).sample(
            frac=1, random_state=10 * (fld + 1)
        )
        df_concat = add_df_sub(df_concat, 10 * (fld + 1))
        df_concat.to_csv(f"{save_dir}/df_fld{fld}.csv", index=False)
        print(df_concat.shape)
        print(df_concat["y"].value_counts())

def save_cleaned_df(df):
    n_folds = 7
    frac_1 = 0.3
    frac_1_factor = 1.2

    for fld in range(n_folds):
        df_concat = pd.concat(
            [
                df[df.y > 0].sample(frac=frac_1, random_state=10 * (fld + 1)),
                df[df.y == 0].sample(
                    n=int(len(df[df.y > 0]) * frac_1 * frac_1_factor),
                    random_state=10 * (fld + 1),
                ),
            ],
            axis=0,
        ).sample(frac=1, random_state=10 * (fld + 1))
        df_concat = add_df_sub(df_concat, 10 * (fld + 1))
        df_concat = clean(df_concat, "text")
        df_concat.to_csv(f"{save_dir}/df_clean_fld{fld}.csv", index=False)
        print(df_concat.shape)
        print(df_concat["y"].value_counts())




def save_ruddit_df():
    # ## Ruddit data
    df_ = pd.read_csv(f"{ruddit_path}/ruddit_with_text.csv")
    print(df_.shape)
    df_ = df_[["txt", "offensiveness_score"]].rename(
        columns={"txt": "text", "offensiveness_score": "y"}
    )
    df_["y"] = (df_["y"] - df_.y.min()) / (df_.y.max() - df_.y.min())
#     df_.y.hist()

    # # Create 3 versions of data
    n_folds = 7
    frac_1 = 0.7
    for fld in range(n_folds):
        print(f"Fold: {fld}")
        df_sampled = df_.sample(frac=frac_1, random_state=10 * (fld + 1))
        df_sampled.to_csv(f"{save_dir}/df2_fld{fld}.csv", index=False)
        print(df_sampled.shape)
        print(df_sampled["y"].value_counts())

save_cleaned_df(df)
save_raw_df(df)
save_ruddit_df()


# Create Sklearn Pipeline with
# TFIDF - Take 'char_wb' as analyzer to capture subwords well
# Ridge - Ridge is a simple regression algorithm that will reduce overfitting

# % of uppercase characters
class LengthUpperTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return sparse.csr_matrix(
            [[sum([1 for y in x if y.isupper()]) / len(x)] for x in X]
        )

    def get_feature_names(self):
        return ["lngth_uppercase"]


fld = 0
df = pd.read_csv(f"{save_dir}/df_clean_fld{fld}.csv")
df.y

def train_on_raw_df():
    val_preds_arr1 = np.zeros((df_val.shape[0], n_folds))
    val_preds_arr2 = np.zeros((df_val.shape[0], n_folds))
    test_preds_arr = np.zeros((df_sub.shape[0], n_folds))

    for fld in range(n_folds):
        print("\n\n")
        print(
            f" ****************************** FOLD: {fld} ******************************"
        )
        df = pd.read_csv(f"{save_dir}/df_fld{fld}.csv")
        print(df.shape)

        tfidf_model = TfidfVectorizer(
                        min_df=3, max_df=0.5, analyzer="char_wb", ngram_range=(3, 5)
                    )
        tfidf_model.fit(df_sub.text)
        embedded_text = tfidf_model.transform(df.text)
        df_nan_removed, indices_list = get_nan_index_list(df)
        embedded_text = delete_from_csr(embedded_text, indices_list)
        ridge_model = Ridge()
        ridge_model.fit(embedded_text, df_nan_removed["y"])

        val_preds_arr1[:, fld] = ridge_model.predict(tfidf_model.transform(df_val["less_toxic"]))
        val_preds_arr2[:, fld] = ridge_model.predict(tfidf_model.transform(df_val["more_toxic"]))
        test_preds_arr[:, fld] = ridge_model.predict(tfidf_model.transform(df_sub["text"]))

        
    return val_preds_arr1, val_preds_arr2, test_preds_arr


def train_on_clean_data():

    val_preds_arr1c = np.zeros((df_val.shape[0], n_folds))
    val_preds_arr2c = np.zeros((df_val.shape[0], n_folds))
    test_preds_arrc = np.zeros((df_sub.shape[0], n_folds))

    for fld in range(n_folds):
        print("\n\n")
        print(
            f" ****************************** FOLD: {fld} ******************************"
        )
        df = pd.read_csv(f"{save_dir}/df_clean_fld{fld}.csv")
        print(df.shape)

        tfidf_model = TfidfVectorizer(
                        min_df=3, max_df=0.5, analyzer="char_wb", ngram_range=(3, 5)
                    )

        tfidf_model.fit(clean(df_sub, "text").text)
        embedded_text = tfidf_model.transform(df.text)
        df_nan_removed, indices_list = get_nan_index_list(df)
        embedded_text = delete_from_csr(embedded_text, indices_list)
        ridge_model = Ridge()
        ridge_model.fit(embedded_text, df_nan_removed["y"])

        val_preds_arr1c[:, fld] = ridge_model.predict(tfidf_model.transform(df_val["less_toxic"]))
        val_preds_arr2c[:, fld] = ridge_model.predict(tfidf_model.transform(df_val["more_toxic"]))
        test_preds_arrc[:, fld] = ridge_model.predict(tfidf_model.transform(df_sub["text"]))

    return val_preds_arr1c, val_preds_arr2c, test_preds_arrc


# ## Ruddit data pipeline
def train_on_ruddit_data():

    val_preds_arr1_ = np.zeros((df_val.shape[0], n_folds))
    val_preds_arr2_ = np.zeros((df_val.shape[0], n_folds))
    test_preds_arr_ = np.zeros((df_sub.shape[0], n_folds))

    for fld in range(n_folds):
        print("\n\n")
        print(
            f" ****************************** FOLD: {fld} ******************************"
        )
        df = pd.read_csv(f"{save_dir}/df2_fld{fld}.csv")
        print(df.shape)

        tfidf_model = TfidfVectorizer(
                        min_df=3, max_df=0.5, analyzer="char_wb", ngram_range=(3, 5)
                    )
        tfidf_model.fit(df_sub.text)
        embedded_text = tfidf_model.transform(df.text)
        df_nan_removed, indices_list = get_nan_index_list(df)
        embedded_text = delete_from_csr(embedded_text, indices_list)
        ridge_model = Ridge()
        ridge_model.fit(embedded_text, df_nan_removed["y"])

        val_preds_arr1_[:, fld] = ridge_model.predict(tfidf_model.transform(df_val["less_toxic"]))
        val_preds_arr2_[:, fld] = ridge_model.predict(tfidf_model.transform(df_val["more_toxic"]))
        test_preds_arr_[:, fld] = ridge_model.predict(tfidf_model.transform(df_sub["text"]))

    return val_preds_arr1_, val_preds_arr2_, test_preds_arr_


def validate_models():

    print(" Toxic CLEAN data ")
    p5 = val_preds_arr1c.mean(axis=1)
    p6 = val_preds_arr2c.mean(axis=1)

    print(f"Validation Accuracy is { np.round((p5 < p6).mean() * 100,2)}")

    print(" Toxic CLEAN data new ")
    p5 = val_preds_arr1c.mean(axis=1)
    p6 = val_preds_arr2c.mean(axis=1)

    print(f"Validation Accuracy is { np.round((p5 < p6).mean() * 100,2)}")

    # # Validate the pipeline

    print(" Toxic data ")
    p1 = val_preds_arr1.mean(axis=1)
    p2 = val_preds_arr2.mean(axis=1)

    print(f"Validation Accuracy is { np.round((p1 < p2).mean() * 100,2)}")

    print(" Ruddit data ")
    p3 = val_preds_arr1_.mean(axis=1)
    p4 = val_preds_arr2_.mean(axis=1)

    print(f"Validation Accuracy is { np.round((p3 < p4).mean() * 100,2)}")

    print(" Toxic CLEAN data ")
    p5 = val_preds_arr1c.mean(axis=1)
    p6 = val_preds_arr2c.mean(axis=1)

    print(f"Validation Accuracy is { np.round((p5 < p6).mean() * 100,2)}")
    return p1, p2, p3, p4, p5, p6


def find_weights():
    wts_acc = []
    for i in range(30, 70, 1):
        for j in range(0, 20, 1):
            w1 = i / 100
            w2 = (100 - i - j) / 100
            w3 = 1 - w1 - w2
            p1_wt = w1 * p1 + w2 * p3 + w3 * p5
            p2_wt = w1 * p2 + w2 * p4 + w3 * p6
            wts_acc.append((w1, w2, w3, np.round((p1_wt < p2_wt).mean() * 100, 2)))
    sorted(wts_acc, key=lambda x: x[3], reverse=True)[:5]

    w1, w2, w3, _ = sorted(wts_acc, key=lambda x: x[2], reverse=True)[0]
    p1_wt = w1 * p1 + w2 * p3 + w3 * p5
    p2_wt = w1 * p2 + w2 * p4 + w3 * p6
    return w1, w2, w3, p1_wt, p2_wt

val_preds_arr1, val_preds_arr2, test_preds_arr = train_on_raw_df()
val_preds_arr1c, val_preds_arr2c, test_preds_arrc = train_on_clean_data()
val_preds_arr1_, val_preds_arr2_, test_preds_arr_ = train_on_ruddit_data()

p1, p2, p3, p4, p5, p6 = validate_models()
w1, w2, w3, p1_wt, p2_wt = find_weights()

# ## Analyze bad predictions
# ### Incorrect predictions with similar scores
# ### Incorrect predictions with different scores

df_val["p1"] = p1_wt
df_val["p2"] = p2_wt
df_val["diff"] = np.abs(p2_wt - p1_wt)
df_val["correct"] = (p1_wt < p2_wt).astype("int")


df_val[df_val.correct == 0].sort_values("diff", ascending=True).head(20)
#### Some of these just look incorrectly tagged
df_val[df_val.correct == 0].sort_values("diff", ascending=False).head(20)

# # Predict on test data
df_sub["score"] = (
    w1 * test_preds_arr.mean(axis=1)
    + w2 * test_preds_arr_.mean(axis=1)
    + w3 * test_preds_arrc.mean(axis=1)
)

# ## Correct the rank ordering
df_sub["score"].count() - df_sub["score"].nunique()
same_score = df_sub["score"].value_counts().reset_index()[:10]
same_score

df_sub[df_sub["score"].isin(same_score["index"].tolist())]
df_sub[["comment_id", "score"]].to_csv("submission.csv", index=False)

