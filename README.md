# Sign Language to Text & Speech (Python)

This small project provides tools to collect hand landmark data (using MediaPipe), train a simple classifier, and run real-time sign recognition with text display and speech output.

Files:
- `collect_data.py` — capture MediaPipe hand landmarks and save labelled CSV rows.
- `train_model.py` — train a scikit-learn classifier and save `model.pkl` and `label_encoder.pkl`.
- `recognize.py` — real-time recognition from webcam; speaks recognized letters/words via offline TTS (`pyttsx3`).
- `requirements.txt` — Python dependencies.

Quick start (Windows PowerShell):

1. Create a virtual env and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Collect data:

```powershell
python collect_data.py
```

Controls in `collect_data.py`:
- Press a letter key (e.g. `A`, `B`, `C`) to set the current label to record.
- Press `r` to toggle recording (it will append landmark rows with the current label).
- Press `q` to quit.

3. Train a model:

```powershell
python train_model.py --input data.csv --output model.pkl
```

4. Run recognition (requires `model.pkl` created by training):

```powershell
python recognize.py --model model.pkl
```

Notes and next steps:
- The repository provides collection and training scaffolding. To get good recognition accuracy you'll need a dataset with many samples per sign and people.
- For production-grade accuracy consider training a deep-learning model on a known ASL dataset and adding temporal models (LSTM/Transformer) for dynamic signs.

License: MIT-style (adapt as you prefer).
