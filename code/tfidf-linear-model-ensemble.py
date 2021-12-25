import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm


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


def train_models(df_concat):

    tfidf_vec = TfidfVectorizer(
        min_df=3, max_df=0.5, analyzer="char_wb", ngram_range=(3, 5)
    )

    vectorized_text = tfidf_vec.fit_transform(df_concat["text"])
    y_col = df_concat["y"]

    # <h1>Fit Ridge</h1>
    model1 = Ridge(alpha=0.5)
    model1.fit(vectorized_text, y_col)

    model2 = Ridge(alpha=1)
    model2.fit(vectorized_text, y_col)

    model3 = Ridge(alpha=2)
    model3.fit(vectorized_text, y_col)

    ridge_m_list = [model1, model2, model3]
    return tfidf_vec, ridge_m_list


def preprocess(df_train):
    # Create a score that measure how much toxic is a comment
    toxicity_dict = {
        "obscene": 0.16,
        "toxic": 0.32,
        "threat": 1.5,
        "insult": 0.64,
        "severe_toxic": 1.5,
        "identity_hate": 1.5,
    }

    # Apply toxicity
    for category in toxicity_dict:
        df_train[category] = df_train[category] * toxicity_dict[category]

    df_train["y"] = df_train.loc[:, "toxic":"identity_hate"].sum(axis=1)
    df_train = df_train.rename(columns={"comment_text": "text"})
    # df_train = df_train.drop_duplicates(subset=["text"])

    tqdm.pandas()
    df_train["text"] = df_train["text"].progress_apply(text_cleaning)
    return df_train


def validate_model(tfidf_vec_list, ridge_m_all):
    df_val = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data.csv")

    # <h2>Text cleaning</h2>
    tqdm.pandas()
    df_val["less_toxic"] = df_val["less_toxic"].progress_apply(text_cleaning)
    df_val["more_toxic"] = df_val["more_toxic"].progress_apply(text_cleaning)

    length = df_val.shape[0]
    p1 = np.array([0.0] * length)
    p2 = np.array([0.0] * length)

    for i, tfidf_vec in enumerate(tfidf_vec_list):
        X_less_toxic = tfidf_vec.transform(df_val["less_toxic"])
        X_more_toxic = tfidf_vec.transform(df_val["more_toxic"])
        for model in ridge_m_all[i * 3 : (i + 1) * 3]:
            p1 += model.predict(X_less_toxic)
            p2 += model.predict(X_more_toxic)
    val_acc = np.round((p1 < p2).mean(), 3)

    print("Validation Accuracy:", val_acc)


from sklearn.model_selection import KFold


def train_model():

    df_train = pd.read_csv(
        "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
    )
    df_train = preprocess(df_train)
    # Undersampling
    min_len = (df_train["y"] > 0).sum()
    df_undersample_not_toxic = df_train[df_train["y"] == 0].sample(
        n=min_len * 2, random_state=402
    )
    df_concat = pd.concat([df_train[df_train["y"] > 0], df_undersample_not_toxic])

    folds = KFold(n_splits=n_folds, shuffle=True, random_state=2021)

    tfidf_vec_list = []
    ridge_m_all = []

    # run model
    for fold_, (trn_idx, val_idx) in enumerate(
        folds.split(df_concat.text, df_concat.y)
    ):
        strLog = "fold {}".format(fold_)
        X_tr, X_val = df_concat.iloc[trn_idx], df_concat.iloc[val_idx]
        y_tr, y_val = df_concat.y.iloc[trn_idx], df_concat.y.iloc[val_idx]
        tfidf_vec, ridge_m_list = train_models(X_tr)
        tfidf_vec_list.append(tfidf_vec)
        ridge_m_all += ridge_m_list
    return tfidf_vec_list, ridge_m_all


def predict_result(tfidf_vec_list, ridge_m_list):
    df_sub = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")
    df_sub["text"] = df_sub["text"].progress_apply(text_cleaning)
    df_sub["score"] = 0
    for i, tfidf_vec in enumerate(tfidf_vec_list):
        # <h2>Prediction</h2>
        X_test = tfidf_vec.transform(df_sub["text"])
        length = df_sub.shape[0]
        p3 = np.array([0.0] * length)
        for model in ridge_m_all[len(ridge_m_list) * i : len(ridge_m_list) * (i + 1)]:
            p3 += model.predict(X_test) / (len(ridge_m_list) * n_folds)
        df_sub["score"] += p3
    df_sub[["comment_id", "score"]].to_csv("../save/submission.csv", index=False)


if __name__ == "__main__":
    n_folds = 5
    tfidf_vec_list, ridge_m_all = train_model()
    validate_model(tfidf_vec_list, ridge_m_all)
