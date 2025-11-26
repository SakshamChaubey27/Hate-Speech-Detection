import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

def main():
    print("Loading dataset...")
    df = pd.read_csv("hate_speech.csv")

    # Show label counts
    print("\nLabel counts:\n", df["label"].value_counts(), "\n")

    X = df["text"]
    y = df["label"]

    # Simple split (no stratify to avoid errors with small data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: TF-IDF + Logistic Regression
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=300))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n===== RESULT =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Interactive testing
    print("\n===== TEST YOUR OWN TEXT =====")
    while True:
        text = input("\nEnter text (or type 'exit'): ")
        if text.lower() == "exit":
            break
        pred = model.predict([text])[0]
        print("Prediction:", "HATE SPEECH ❌" if pred == 1 else "NOT HATE SPEECH ✅")

if __name__ == "__main__":
    main()
