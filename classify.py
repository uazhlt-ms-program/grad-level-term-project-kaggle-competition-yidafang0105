import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print("train shape:", train_df.shape)
print("test shape:", test_df.shape)
print(train_df["LABEL"].value_counts())

# quick sanity check
print(train_df.head(2))

def clean_text(text):
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s'.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

print("cleaning...")
train_df["clean"] = train_df["TEXT"].apply(clean_text)
test_df["clean"] = test_df["TEXT"].apply(clean_text)

# check one sample to make sure cleaning works
print(train_df["clean"].iloc[0][:200])

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100000,
        sublinear_tf=True,
        min_df=2,
    )),
    ("clf", LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="saga",
        random_state=42,
    )),
])

# split to check performance before final submission
X_train, X_val, y_train, y_val = train_test_split(
    train_df["clean"], train_df["LABEL"],
    test_size=0.2, random_state=42, stratify=train_df["LABEL"]
)

print("fitting on train split...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_val)
macro_f1 = f1_score(y_val, y_pred, average="macro")
print("val macro f1:", macro_f1)
print(classification_report(y_val, y_pred, target_names=["not-review", "positive", "negative"]))

# retrain on everything before predicting test
print("retraining on full data...")
pipeline.fit(train_df["clean"], train_df["LABEL"])

preds = pipeline.predict(test_df["clean"])

out = pd.DataFrame({"ID": test_df["ID"], "LABEL": preds})
out.to_csv("submission.csv", index=False)
print("saved submission.csv, rows:", len(out))
