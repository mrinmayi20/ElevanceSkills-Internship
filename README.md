# ElevanceSkills AI/ML Internship
## Project Overview
This repository was developed as part of the AI/ML Internship at ElevanceSkills Technology Private Limited.
The repository combines six machine learning tasks spanning computer vision, audio processing, and multi-signal decision logic. Each task applies deep learning models (CNNs, and in one case YOLOv8) to a distinct real-world scenario, paired with a graphical interface (mostly Gradio) for interactive use.

## Project Objectives
The main objectives of this project are to:
- Train and evaluate CNN-based models for image, video, and audio classification tasks.
- Apply rule-based logic on top of trained models where a required behaviour has no direct labelled dataset (e.g., hair length, dress colour, age-gated overrides).
- Build interactive GUIs for both image upload and real-time video/webcam use.
- Implement time-restricted and condition-gated system behaviour per task requirements.
- Evaluate each model with appropriate metrics (accuracy, MAE, F1-score, confusion matrices) and document results transparently, including known limitations.
- Develop practical skills in end-to-end ML project delivery - from dataset selection through to a working, demoable interface.

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- YOLOv8 (Ultralytics)
- Librosa
- NumPy / Pandas
- Matplotlib / Scikit-Learn
- Gradio
- openpyxl

## Datasets
- ASL Alphabet (Kaggle - grassknoted/asl-alphabet)
- Age, Gender and Ethnicity (Face Data) CSV (Kaggle - nipunarora8)
- UTKFace (Kaggle - jangedoo/utkface-new)
- FairFace (Kaggle - aibloy/fairface)
- FER2013 (Kaggle)
- Fashion Product Images Dataset (Kaggle - paramaggarwal)
- Vehicle Colour Recognition Dataset - VCoR (Kaggle)
- CREMA-D (Kaggle - ejlok1/cremad)
- RAVDESS Emotional Speech Audio (Kaggle - uwrfkaggler)

## Project Tasks
### Task 1 - Sign Language Detection
Trains a CNN to recognize ASL alphabet gestures and combine predicted letters into known words.

Key features include:
- Image upload and real-time webcam prediction
- Confidence-thresholded, temporal-smoothed live predictions
- Known-word recognition from spelled letters
- Time-restricted operation (6 PM-10 PM)
- **Result:** 99.94% validation accuracy across 29 classes

### Task 2 - Senior Citizen Identification
Detects multiple people in video/webcam feeds, predicts age and gender, and flags seniors (60+).

Key features include:
- Multi-person face detection and frame-to-frame tracking
- Age and gender prediction per tracked visitor
- Automated CSV/Excel visit logging
- **Result:** 85.2% gender accuracy, 7.44-year age MAE, 93.7% senior-flag accuracy

### Task 3 - Long Hair Identification
Applies an age-gated rule where hair length overrides gender prediction for ages 20-30 only.

Key features include:
- Age and gender CNNs
- Rule-based hair-length detection (dark-pixel ratio)
- Age-gated override logic, verified against real and hand-picked test cases
- **Result:** 86% gender accuracy, 9.79-year age MAE

### Task 4 - Car Colour Detection and Traffic Analysis
Detects vehicles and pedestrians in traffic images and highlights vehicles by colour.

Key features include:
- YOLOv8 vehicle/pedestrian detection
- CNN-based vehicle colour classification
- Colour-conditional bounding boxes (blue -> red box, others -> blue box)
- **Result:** 78.86% colour classification accuracy

### Task 5 - Age and Emotion Detection through Voice
Predicts age and gender from voice notes, gated by a male-only requirement and a senior-citizen emotion check.

Key features include:
- Mel-spectrogram-based CNNs for gender, age, and emotion
- Automatic rejection of non-male voices
- Emotion detection only for predicted seniors (60+)
- **Result:** ~90% gender accuracy, 91.5% senior-threshold accuracy, ~46-50% emotion accuracy

### Task 6 - Nationality Detection Model
Predicts nationality and emotion from an image, with conditional age/dress-colour output.

Key features include:
- Nationality (4-class) and age CNN, plus a separate emotion CNN
- Rule-based dress-colour matching from a real-data colour reference table
- Nationality-dependent conditional output logic
- Generalization tested on a held-out independent dataset (FairFace)

## Common GUI Features
Across the six tasks, the interfaces provide:
- Image upload and/or real-time webcam/video input
- Interactive Gradio-based interface
- Preview of uploaded input
- Model prediction output with supporting detail (confidence, intermediate results)
- Time-based or condition-based availability where required by the task

## Project Structure
ElevanceSkills-AI-ML-Internship/
│
├── sign-language-detection/
├── senior-citizen-identification/
├── long-hair-identification/
├── car-colour-detection/
├── age-emotion-voice-detection/
├── nationality-detection/
│
└── README.md
Each task folder contains its own notebook, README, and report with full setup/usage instructions, dataset details, model architecture, and documented limitations.
