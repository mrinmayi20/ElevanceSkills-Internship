# Nationality Detection Model

## Project Overview

This project implements an image-based Nationality, Emotion, Age, and Dress Colour Prediction System using Deep Learning and Computer Vision.

The system can:

- Predict a person's nationality category from their image
- Predict the person's emotion
- Predict age (for Indian and USA nationality)
- Predict dress colour (for Indian and African nationality)
- Display results through a Gradio GUI

## Problem Statement

The objective is to develop a machine learning system that takes a person's image as input and produces the required predictions, with the extra outputs conditional on the predicted nationality.

### Decision Rule

For predicted nationality **Indian**:
- Show age, dress colour, and emotion

For predicted nationality **USA**:
- Show age and emotion

For predicted nationality **African**:
- Show dress colour and emotion

For **other** nationality categories:
- Show only nationality and emotion

## Dataset

Four datasets are used, each for a different part of the pipeline.

### UTKFace Dataset
Age, gender, and ethnicity labels are extracted directly from the image filenames. Ethnicity labels are remapped onto the four nationality buckets this task requires:

- White → USA
- Black → African
- Asian → Other
- Indian → Indian
- Others → Other

This is the dataset the nationality + age CNN is trained on.

### FairFace Dataset
A separately collected, more demographically balanced face dataset with its own race labels. It is never used during training — it is held out to test whether the nationality model generalises to faces it has never seen, rather than just memorising UTKFace.

### FER2013 Dataset
Used for emotion classification. Seven emotion classes:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

### Fashion Product Images Dataset
Real clothing product photos labelled with actual colour names. Instead of hand-picking a palette of RGB values, colour names for dress-colour prediction are built directly from this dataset by averaging pixel colours from real labelled photos per colour category.

### Dataset Caps

For faster training, both face datasets are capped per class rather than using the full dataset:

| Dataset | Cap per class | Classes |
|---|---|---|
| UTKFace | up to 1,500 images | 5 raw race labels |
| FER2013 | up to 4,000 images | 7 emotions |

Both are split 85% train / 15% validation.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy / Pandas
- Matplotlib
- Scikit-learn
- Gradio

## Methodology

### Step 1 — Data Preprocessing
Face crops are resized to 64×64 (nationality/age model) or 48×48 grayscale (emotion model). Pixel values are normalised to 0–1. Age targets are scaled by /100 during training for stability.

### Step 2 — Nationality + Age CNN
A shared CNN backbone with two output heads.

Architecture:
- Conv2D — 32 filters + BatchNorm + MaxPooling
- Conv2D — 64 filters + BatchNorm + MaxPooling
- Conv2D — 128 filters + BatchNorm + MaxPooling
- Flatten
- Dense — 128
- Dropout — 0.3
- Two output heads: Softmax(4) for nationality, Linear(1) for age

Training:
- Optimizer: Adam
- Losses: Sparse Categorical Crossentropy (nationality) + Mean Squared Error (age)
- Epochs: 10
- Batch size: 32

### Step 3 — Emotion CNN
A separate CNN for 7-class emotion classification.

Architecture:
- Conv2D — 32 filters + BatchNorm + MaxPooling
- Conv2D — 64 filters + BatchNorm + MaxPooling
- Conv2D — 128 filters + BatchNorm + MaxPooling
- Flatten
- Dense — 256
- Dropout — 0.4
- Softmax(7) output

Training:
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 25
- Batch size: 64

### Step 4 — Face Detection
OpenCV's frontal-face Haar cascade detects and crops the face before either model sees the image.

### Step 5 — Dress Colour Detection
A region just below the detected face (centered, roughly face-width) is averaged to one representative RGB colour. That colour is matched in two stages:
1. Classified into a main colour family (Red, Orange, Yellow, Green, Blue, Purple, Pink, Black, White, Gray) by hue angle.
2. Matched to the closest dataset-derived shade **within that same family only**, using the colour names built from the Fashion Product Images Dataset.

No face → dress colour is not evaluated. No clothing visible in the crop → result may be unreliable (this is a heuristic, not a learned feature).

### Step 6 — Final Decision
The predicted nationality controls which extra fields (age, dress colour) are shown, per the Decision Rule above.

### Step 7 — Gradio GUI
The user uploads a person's image and receives:
- Predicted Nationality
- Predicted Emotion
- Age (if applicable)
- Dress Colour (if applicable)

## Results

Both models are evaluated on held-out data — not the images used for training.

| Metric | Evaluated on |
|---|---|
| Nationality accuracy | Held-out UTKFace validation split |
| Age MAE | Held-out UTKFace validation split |
| Emotion accuracy | Held-out FER2013 validation split |
| Nationality accuracy (independent) | FairFace — never touched during training |
| Dress colour accuracy | Held-out Fashion Product Images samples |

Exact numbers vary slightly between runs (random train/validation split, random weight initialisation) — run the notebook's Accuracy Summary cell to see the current numbers for a given run, rather than relying on figures written here.

The FairFace number is the more meaningful nationality metric, since it tests real generalisation rather than memorisation of UTKFace-specific patterns.

## Limitations

- Nationality can't really be predicted from a face — it's approximated from ethnicity labels, not a real nationality detector.
- The White → USA mapping is weak — most white people aren't American, and plenty of Americans aren't white.
- "African" is a continent, not an ethnicity — same problem as above.
- Both models are trained on a limited subset of data and epochs to save time, which caps their accuracy.
- FER2013 has imbalanced classes (e.g. "disgust" has very few images), which can bias the emotion model.
- Dress colour is a simple heuristic — it only works when clothing is actually visible in the photo.
- Treat all predictions as demo output, not real judgments about a person.

## Future Improvements

- Train on the full datasets with more epochs.
- Compare against a pretrained model (transfer learning).
- Fix FER2013's class imbalance properly.
- Improve dress colour detection with real clothing segmentation.
- Test on more varied real-world photos.
- Add better error handling to the Gradio interface.

## Conclusion

This project builds an end-to-end image-based pipeline: data preprocessing, two CNNs trained from scratch (nationality + age on UTKFace, emotion on FER2013), a rule-based dress-colour detector built entirely from real labelled clothing data (no hand-picked colours), held-out evaluation across all four attached datasets, and a Gradio interface for interactive use. The nationality mapping is checked against both UTKFace's own held-out split and FairFace, a completely independent dataset never seen during training, to distinguish genuine generalisation from memorisation. No pretrained/transfer-learning model was benchmarked against the custom CNNs in this version — both are trained entirely from scratch, which is listed as a future improvement rather than something already demonstrated here.

## Project Structure

```
Nationality-Detection/
│
├── nationality-detection.ipynb
├── README.md
├── requirements.txt
└── images/
```

## How to Run

1. Open the notebook.
2. Attach the four datasets: UTKFace, FairFace, FER2013, Fashion Product Images Dataset.
3. Set the dataset paths (fill in the `MANUAL_*_ROOT` variables if auto-detection doesn't find a dataset).
4. Run the preprocessing cells.
5. Train the nationality + age model, then the emotion model.
6. Run the spot-check and held-out evaluation cells.
7. Launch the Gradio GUI.
8. Upload an image and view the result.

## Author

**Mrinmayi Kunkaliencar**
Data Science & Machine Learning Project
