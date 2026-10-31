from pathlib import Path
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset" / "email_dataset.csv"
OUT = BASE / "models"

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA).dropna(subset=["text", "label"])
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV needs 'text' and 'label' columns.")
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"])
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=8000)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    model = MultinomialNB(alpha=0.5).fit(Xtr, y_train)
    pred = model.predict(Xte)
    OUT.mkdir(exist_ok=True)
    joblib.dump(model, OUT / "email_model.pkl")
    joblib.dump(vec, OUT / "tfidf_vectorizer.pkl")
    print("\\n=== Held-out sample evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, pred):.2%}")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix class order:", list(model.classes_))
    print(confusion_matrix(y_test, pred, labels=model.classes_))
    print("\\nSaved model in models/. This small illustrative dataset does not prove real-world security.")
if __name__ == "__main__":
    main()
