import os
import zipfile
import urllib.request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ==================================================
# 1. DOWNLOAD DATASET
# ==================================================

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

zip_file = "sms_spam.zip"
data_file = "SMSSpamCollection"

if not os.path.exists(data_file):

    print("Downloading SMS Spam Collection...")

    urllib.request.urlretrieve(url, zip_file)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(".")

    print("Dataset downloaded successfully.")


# ==================================================
# 2. LOAD DATASET
# ==================================================

df = pd.read_csv(
    data_file,
    sep="\t",
    header=None,
    names=["label", "text"]
)

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nDataset size:", len(df))

print("\nClass Distribution:")
print(df["label"].value_counts())


# ==================================================
# 3. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==================================================
# 4. TF-IDF FEATURE EXTRACTION
# ==================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF features:", X_train_tfidf.shape[1])


# ==================================================
# 5. TRAIN LOGISTIC REGRESSION
# ==================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)


# ==================================================
# 6. MODEL EVALUATION
# ==================================================

probabilities = model.predict_proba(
    X_test_tfidf
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


print("\n")
print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Genuine", "Spam"],
        zero_division=0
    )
)


# ==================================================
# 7. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Genuine", "Spam"]
)

disp.plot()

plt.title(
    "Spam Email Classifier - Confusion Matrix"
)

plt.show()


# ==================================================
# 8. LIVE EMAIL CLASSIFIER
# ==================================================

def classify_email(email, threshold=0.5):

    email_vector = vectorizer.transform(
        [email]
    )

    spam_probability = model.predict_proba(
        email_vector
    )[0][1]

    if spam_probability >= threshold:
        result = "SPAM"
    else:
        result = "GENUINE"

    print("\n")
    print("-" * 50)

    print("EMAIL:")
    print(email)

    print(
        f"\nSpam Probability: "
        f"{spam_probability:.2%}"
    )

    print(
        f"Decision Threshold: "
        f"{threshold:.2%}"
    )

    print(
        f"\nPrediction: {result}"
    )

    print("-" * 50)


# ==================================================
# 9. LIVE DEMO
# ==================================================

print("\n")
print("=" * 50)
print("SPAM EMAIL DETECTOR")
print("=" * 50)

while True:

    email = input(
        "\nEnter an email/message "
        "(or type 'exit'): "
    )

    if email.lower() == "exit":
        break

    threshold_input = input(
        "Enter threshold (0-1), "
        "default 0.5: "
    )

    if threshold_input == "":
        threshold = 0.5
    else:
        threshold = float(
            threshold_input
        )

    classify_email(
        email,
        threshold
    )