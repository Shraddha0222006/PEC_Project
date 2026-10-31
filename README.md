# AI Email Security — Enhanced Student Demo

## Features
- Spam / phishing / likely-safe classification
- Suspicious phrase and link indicators
- Responsive blue-and-white interface
- Model confidence display (not a real-world risk probability)
- Train/test accuracy, precision, recall, F1-score and confusion matrix

## Windows PowerShell setup
Open this folder in VS Code, then run:
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python app.py
```
If `venv` already exists in this folder, do not create it again. Open the local URL shown in the terminal, usually http://127.0.0.1:5000.

## Limitations
The included CSV is a small illustrative dataset with templated variations, not a real-world corpus. Metrics may be optimistic and do not establish real-world accuracy. Do not paste private emails, passwords or OTPs. “Likely Safe” is not a guarantee.
