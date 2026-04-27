# ASL Sign Language App

Real-time American Sign Language alphabet recognition using a webcam. MediaPipe detects 21 hand landmarks per frame, which are normalized and enriched with finger joint angles, and a Random Forest classifier predicts which ASL letter is being signed.

---

## Demo

*Video coming soon*

---

## Features

- Real-time ASL alphabet recognition (A-Z) via webcam
- Hand landmark visualization with skeleton overlay
- Probability smoothing across a 7-frame buffer for stable predictions
- Stable-frame debounce system to prevent flickering letters being added mid-transition
- On-screen sentence builder with keyboard shortcuts

**Keyboard Shortcuts**

| Key | Action |
|-----|--------|
| `SPACE` | Add a space |
| `BACKSPACE` | Delete last character |
| `C` | Clear sentence |
| `Q` | Quit |

---

## How It Works

1. MediaPipe detects 21 hand landmarks per frame
2. Landmarks are normalized (wrist-relative + scale normalization)
3. 9 finger joint bend angles are computed and appended to the feature vector (72 features total)
4. A Random Forest classifier predicts the letter
5. Predictions are smoothed across 7 frames and must hold stable for 8 consecutive frames before being added to the sentence

---

## Setup & Installation

### Requirements

- Python 3.9+
- Webcam

### 1. Clone the repository

```bash
git clone https://github.com/a-arya1/asl-sign-language-app.git
cd asl-sign-language-app
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the pretrained model

The model file is too large to host on GitHub. Download it from Google Drive and place it in the root of the project folder:

[Download hand_gesture_model.joblib](https://drive.google.com/file/d/1Cv8PrXy5M9PfmDbuoKZz91maVrv6jJm_/view?usp=sharing)

Your project folder should look like this before running:
asl-sign-language-app/
├── hand_gesture_model.joblib   ← place it here
├── hand_tracker.py
├── ...

### 5. Run the app

```bash
python hand_tracker.py
```

---

## Retraining the Model

You only need to do this if you want to add your own training data or retrain from scratch.

### Datasets used

- [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — place at `archive/asl_alphabet_train/asl_alphabet_train`
- [Mendeley ASL Dataset](https://data.mendeley.com) — place at your local path and update `mendeley_dataset` in `processdata.py`
- Personal webcam data collected using `collect_data.py`

### Steps

**1. (Optional) Collect your own data**

```bash
python collect_data.py
```

Press a letter key to start collecting frames for that letter. The script auto-collects while your hand is detected and appends to `handsData.csv`.

**2. Process the datasets into a CSV**

```bash
python processdata.py
```

**3. Retrain the model**

```bash
python model.py
```

> **Important:** Any time `normalize_data.py` is changed, all three steps above must be re-run in order to keep the pipeline in sync.

---

## File Structure
asl-sign-language-app/
├── hand_tracker.py            # Live webcam recognition app
├── model.py                   # Trains and saves the Random Forest classifier
├── normalize_data.py          # Landmark normalization and angle feature extraction
├── processdata.py             # Builds training CSV from image datasets
├── collect_data.py            # Webcam tool for collecting personal training data
├── hand_landmarker.task       # MediaPipe hand landmark model
└── hand_gesture_model.joblib  # Pretrained model (download from Google Drive above)

---

## Known Limitations

- **E / S / C and M / N confusion** — these letters look very similar as static poses. Recording personal training data for these letters helps most.
- **J and Z not supported** — these require motion and cannot be recognized by a single-frame model.
- **Lighting sensitivity** — the model was trained on studio images. Poor webcam lighting will lower confidence. Collecting personal data in your normal environment helps.

---

## Roadmap

- [ ] **Phase 1 — Static word signs:** Add common ASL words (HELP, YES, NO, PLEASE, etc.) as new classes in the existing model
- [ ] **Phase 2 — MLP classifier:** Replace Random Forest with a neural network for a higher accuracy ceiling (target 90-97%)
- [ ] **Phase 3 — Motion sign recognition:** LSTM sequence model reading 30-frame windows to recognize signs that involve movement (WATER, MILK, THANK YOU, etc.)

---

## Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) by Google for hand landmark detection
- [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) by Akash Nagaraj
- Mendeley ASL Sign Alphabet Dataset