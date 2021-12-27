import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

from tqdm.auto import tqdm

import matplotlib.pyplot as plt


def text_cleaning(text):
    """
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis
    
    text - Text piece to be cleaned.
    """
    template = re.compile(r"https?://\S+|www\.\S+")  # Removes website links
    text = template.sub(r"", text)

    soup = BeautifulSoup(text, "lxml")  # Removes HTML tags
    only_text = soup.get_text()
    text = only_text

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    text = re.sub(r"[^a-zA-Z\d]", " ", text)  # Remove special Charecters
    text = re.sub(" +", " ", text)  # Remove Extra Spaces
    text = text.strip()  # remove spaces at the beginning and at the end of string

    return text


from sklearn.metrics import mean_squared_error


def validate(df_valid, tfidf_vec, ridge_m_list):
    vectorized_text = tfidf_vec.transform(df_valid["text"])
    y_col = df_valid["y"]
    pred_col = np.array([0.0] * df_valid.shape[0])
    for model in ridge_m_list:
        pred_col += model.predict(vectorized_text) / len(ridge_m_list)
    RMSE = mean_squared_error(y_col, pred_col) ** 0.5
    return np.round(RMSE, 3)


def train_models(df_concat):

    tfidf_vec = TfidfVectorizer(
        min_df=3, max_df=0.5, analyzer="char_wb", ngram_range=(3, 5)
    )

    vectorized_text = tfidf_vec.fit_transform(df_concat["text"])
    y_col = df_concat["y"]

    # vectorized_text = normalize(vectorized_text, norm="l2", axis=1)
    # <h1>Fit Ridge</h1>

    # model = LogisticRegression(max_iter=200)
    # y_col = (y_col > 0).astype(int)
    # model.fit(vectorized_text, y_col)
    # ridge_m_list = [model]
    model1 = Ridge(alpha=0.5)
    model1.fit(vectorized_text, y_col)

    model2 = Ridge(alpha=1)
    model2.fit(vectorized_text, y_col)

    model3 = Ridge(alpha=2)
    model3.fit(vectorized_text, y_col)

    ridge_m_list = [model1, model2, model3]
    return tfidf_vec, ridge_m_list


def merge_cols(df_train, column_list):

    # Apply toxicity
    for category in toxicity_dict:
        df_train[category] = df_train[category] * toxicity_dict[category]
    df_train["y"] = df_train.loc[:, column_list].sum(axis=1)
    # set values to 1
    # df_train.loc[df_train[df_train["y"] > 0].index, "y"] = 1
    # df_train = df_train.drop_duplicates(subset=["text"])

    return df_train


def validate_model(df_val, tfidf_vec_list, ridge_m_all):

    val_length = df_val.shape[0]
    p1 = np.array([0.0] * val_length)
    p2 = np.array([0.0] * val_length)

    for i, tfidf_vec in enumerate(tfidf_vec_list):
        X_less_toxic = tfidf_vec.transform(df_val["less_toxic"])
        X_more_toxic = tfidf_vec.transform(df_val["more_toxic"])
        for model in ridge_m_all[
            i
            * int(len(ridge_m_all) / n_folds) : (i + 1)
            * int(len(ridge_m_all) / n_folds)
        ]:
            p1 += model.predict(X_less_toxic)
            p2 += model.predict(X_more_toxic)
            # p1 += model.predict(X_less_toxic)[:, 1]
            # p2 += model.predict(X_more_toxic)[:, 1]
    return p1, p2


from sklearn.model_selection import KFold


# df_train.to_csv("../df_train.csv", index=False)


def get_trained_models(df_train):

    # Undersampling
    min_len = (df_train["y"] > 0).sum()
    df_undersample_not_toxic = df_train[df_train["y"] == 0].sample(
        n=min_len * 2, random_state=402
    )
    df_concat = pd.concat([df_train[df_train["y"] > 0], df_undersample_not_toxic])

    folds = KFold(n_splits=n_folds, shuffle=True, random_state=2021)

    tfidf_vec_list = []
    ridge_m_all = []
    RMSE_list = []

    # run model
    for fold_, (trn_idx, val_idx) in enumerate(
        folds.split(df_concat.text, df_concat.y)
    ):
        X_tr, X_val = df_concat.iloc[trn_idx], df_concat.iloc[val_idx]
        tfidf_vec, ridge_m_list = train_models(X_tr)
        RMSE = validate(X_val, tfidf_vec, ridge_m_list)
        RMSE_list.append(RMSE)
        tfidf_vec_list.append(tfidf_vec)
        ridge_m_all += ridge_m_list
    print("RMSE:", np.round(np.mean(RMSE_list), 3))
    return tfidf_vec_list, ridge_m_all


def predict_result(tfidf_vec_list, ridge_m_all):

    for i, tfidf_vec in enumerate(tfidf_vec_list):
        # <h2>Prediction</h2>
        X_test = tfidf_vec.transform(df_sub["text"])
        length = df_sub.shape[0]
        p3 = np.array([0.0] * length)
        for model in ridge_m_all[
            i
            * int(len(ridge_m_all) / n_folds) : (i + 1)
            * int(len(ridge_m_all) / n_folds)
        ]:
            # p3 += model.predict(X_test)[:, 1] / (n_folds * len(ridge_m_all))
            p3 += model.predict(X_test) / (n_folds * len(ridge_m_all))
        df_sub["score"] += p3
    # df_sub[["comment_id", "score"]].to_csv("../save/submission.csv", index=False)
    return df_sub


def load_ruddit_data():
    df_ = pd.read_csv("../input/ruddit-jigsaw-dataset/ruddit_with_text.csv")

    tqdm.pandas()
    df_["txt"] = df_["txt"].progress_apply(text_cleaning)
    df_ = df_[["txt", "offensiveness_score"]].rename(
        columns={"txt": "text", "offensiveness_score": "y"}
    )
    df_["y"] = (df_["y"] - df_.y.min()) / (df_.y.max() - df_.y.min())
    df_ = df_[["text", "y"]]
    return df_


if __name__ == "__main__":

    # Create a score that measure how much toxic is a comment
    toxicity_dict = {
        "obscene": 0.16,
        "toxic": 0.32,
        "threat": 1.5,
        "insult": 0.64,
        "severe_toxic": 1.5,
        "identity_hate": 1.5,
    }

    # toxicity_dict = {
    #     "obscene": 1,
    #     "toxic": 1,
    #     "threat": 1,
    #     "insult": 1,
    #     "severe_toxic": 2,
    #     "identity_hate": 1,
    # }

    n_folds = 5
    df_train = pd.read_csv(
        "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
    )
    df_train = df_train.rename(columns={"comment_text": "text"})
    tqdm.pandas()
    df_train["text"] = df_train["text"].progress_apply(text_cleaning)

    df_val = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data.csv")
    tqdm.pandas()
    df_val["less_toxic"] = df_val["less_toxic"].progress_apply(text_cleaning)
    df_val["more_toxic"] = df_val["more_toxic"].progress_apply(text_cleaning)

    df_sub = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")
    tqdm.pandas()
    df_sub["text"] = df_sub["text"].progress_apply(text_cleaning)
    df_sub["score"] = 0

    column_list_of_list = [
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",]
    ]

    train_length = df_train.shape[0]
    val_length = df_val.shape[0]
    sub_length = df_sub.shape[0]
    p1_save = np.array([0.0] * val_length)
    p2_save = np.array([0.0] * val_length)
    score_save = np.array([0.0] * sub_length)

    for column_list in column_list_of_list:

        df_train = merge_cols(df_train, column_list)
        df_train = df_train[["text", "y"]]
        # df_train_ = load_ruddit_data()
        # df_train = pd.concat([df_train, df_train_],axis=0)

        (tfidf_vec_list, ridge_m_all) = get_trained_models(df_train)

        p1, p2 = validate_model(df_val, tfidf_vec_list, ridge_m_all)
        df_sub = predict_result(tfidf_vec_list, ridge_m_all)

        p1_save += p1
        p2_save += p2
        score_save += df_sub.score

    val_acc = np.round((p1_save < p2_save).mean(), 3)
    print("Validation Accuracy:", val_acc)
    score_save = MinMaxScaler().fit_transform(np.array(score_save).reshape(-1, 1))
    df_sub.score = score_save
    df_sub[["comment_id", "score"]].to_csv("./submission.csv", index=False)

# df_train.columns

# np.sum(df_train.toxic)
# np.sum(df_train.severe_toxic)

# df_train[df_train.severe_toxic == 1][["toxic", "severe_toxic"]]
