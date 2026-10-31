from flask import Flask, render_template, request
from pathlib import Path
import re, joblib

app = Flask(__name__)
BASE = Path(__file__).resolve().parent
MODEL = BASE / "models" / "email_model.pkl"
VECTORIZER = BASE / "models" / "tfidf_vectorizer.pkl"

def load_files():
    if not MODEL.exists() or not VECTORIZER.exists():
        return None, None
    return joblib.load(MODEL), joblib.load(VECTORIZER)

def indicators(text):
    lower = text.lower()
    phrases = ["verify your account", "urgent action", "account suspended", "click here",
               "claim your prize", "send otp", "share your password", "confirm your password",
               "reset your password", "payment details", "account will be locked"]
    found = [f"Suspicious phrase: “{p}”" for p in phrases if p in lower]
    links = re.findall(r"(?:https?://|www\\.)\\S+", text, re.I)
    if links:
        found.append(f"Contains {len(links)} web link(s). Check the destination carefully.")
    if re.search(r"\\b(otp|password|pin|passcode)\\b", lower):
        found.append("Mentions sensitive information such as an OTP, password, PIN or passcode.")
    if not found:
        found.append("No common warning phrases detected. This does not prove the email is safe.")
    return found

@app.route("/", methods=["GET", "POST"])
def index():
    result, warning_list, error = None, [], None
    email_text = ""
    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()
        if not email_text:
            error = "Please paste an email message before analyzing it."
        else:
            model, vectorizer = load_files()
            if model is None:
                error = "Model files missing. Run python train_model.py first."
            else:
                vector = vectorizer.transform([email_text])
                predicted = model.predict(vector)[0]
                confidence = round(float(max(model.predict_proba(vector)[0])) * 100)
                labels = {
                    "spam": ("SPAM", "This text resembles spam examples.", "spam"),
                    "phishing": ("PHISHING RISK", "This text resembles phishing examples. Do not click links or share credentials.", "danger"),
                    "likely_safe": ("LIKELY SAFE", "The model did not find a strong match to its spam/phishing examples. Verify the sender independently.", "safe")
                }
                label, explanation, css_class = labels.get(predicted, ("NEEDS REVIEW", "Review this message carefully.", "review"))
                result = {"label": label, "explanation": explanation, "confidence": confidence, "class": css_class}
                warning_list = indicators(email_text)
    return render_template("index.html", result=result, indicators=warning_list, error=error, email_text=email_text)

if __name__ == "__main__":
    app.run(debug=True)
