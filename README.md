# Speech Emotion Recognition

A machine learning application that detects human emotions from audio files using MFCC features and a Neural Network classifier.

---

## What It Does

- Loads audio files from the RAVDESS dataset
- Extracts MFCC (Mel Frequency Cepstral Coefficients) features from each audio file
- Trains a Neural Network (MLP) to classify emotions
- Predicts the emotion of any new audio file with confidence scores
- Achieves **75% accuracy** on the test set

---

## Emotions Detected

| Code | Emotion |
|---|---|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

---

## How It Works

The app works in four main steps:

**1. Feature Extraction**
Each audio file is loaded using Librosa. 40 MFCC features are extracted from the audio signal. MFCCs capture the tonal and frequency patterns of speech — essentially how the voice sounds. The mean of each MFCC is taken to create a fixed-size feature vector of 40 values.

**2. Dataset Loading**
Audio files are loaded from the RAVDESS dataset folder structure. The emotion label is extracted directly from the filename (the third segment of the filename contains the emotion code).

**3. Model Training**
Features are normalized using StandardScaler. An MLP (Multi-Layer Perceptron) Neural Network with three hidden layers (300, 200, 100 neurons) is trained on 80% of the data using the Adam optimizer and tanh activation.

**4. Prediction**
For any new audio file, features are extracted, scaled, and passed to the trained model. The model returns the predicted emotion along with confidence scores for all emotions.

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download the RAVDESS dataset from Kaggle:
[https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio]

Place the extracted folder as `audio_speech_actors_01-24/` in the project root.

**4. Run the notebook**
```bash
jupyter notebook emotion_recognition.ipynb
```

---

## Requirements

```
librosa
numpy
scikit-learn
jupyter
```

Install all:
```bash
pip install librosa numpy scikit-learn jupyter
```

---

## Model Details

| Parameter | Value |
|---|---|
| Model | MLPClassifier |
| Hidden Layers | 300 → 200 → 100 neurons |
| Activation | tanh |
| Optimizer | Adam |
| Max Iterations | 500 |
| Test Split | 20% |
| Features | 40 MFCC means |
| **Accuracy** | **75%** |

## Dataset

This project uses the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

---
## Project Demo:- https://drive.google.com/file/d/1UZydJzqs8Sxz6L7EHfxpah-k30khH708/view?usp=drive_link
