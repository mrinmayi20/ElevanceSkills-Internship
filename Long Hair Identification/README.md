# Long Hair Identification System

## Project Overview
This project implements an image-based Age-Gender and Long Hair Identification System using Deep Learning and Computer Vision.

The system can:
- Predict age from a person's image
- Predict gender using a CNN
- Detect hair length using OpenCV
- Apply an age-based gender decision rule
- Display results through a Gradio GUI

## Problem Statement
The objective is to develop a machine learning system capable of predicting age and gender from facial images and using hair-length information for a task-specific gender decision.

### Decision Rule
For predicted age **20–30**:
- Long hair → Female
- Short hair → Male

For ages outside 20–30:
- Use the CNN gender prediction.

## Dataset

### UTKFace Dataset
The project uses the UTKFace dataset. Age and gender labels are extracted from the image filenames.

Gender labels:
- `0` → Male
- `1` → Female

For faster training, **4,000 images** were used.

### Dataset Split
- Total images: 4,000
- Training images: 3,200
- Testing images: 800

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Gradio

## Methodology

### Step 1 — Data Preprocessing
Images are resized to `64 × 64` pixels. Age and gender labels are extracted from filenames. The data is split using an 80/20 train-test split.

### Step 2 — Gender CNN
A CNN is trained for binary gender classification.

Architecture:
- Conv2D — 32 filters
- MaxPooling
- Conv2D — 64 filters
- MaxPooling
- Conv2D — 128 filters
- MaxPooling
- Flatten
- Dense — 128
- Dropout — 0.3
- Sigmoid output

Training:
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Epochs: 10
- Batch size: 32

### Step 3 — Age CNN
A CNN-based regression model predicts continuous age.

- Optimizer: Adam
- Loss: Mean Squared Error
- Metric: MAE
- Epochs: 10
- Batch size: 32

### Step 4 — Hair Detection
OpenCV's frontal-face Haar cascade detects the face. A region below the detected face is analysed using grayscale pixel intensity.

- Dark-pixel ratio > 0.25 → Long
- Otherwise → Short
- No detected face → Unknown

### Step 5 — Final Decision
The predicted age controls whether the hair-based rule is applied.

### Step 6 — Gradio GUI
The user uploads a person's image and receives:
- Predicted Age
- CNN Gender
- Hair Length
- Age Gate Status
- Final Gender

## Results

| Metric | Result |
|---|---:|
| Images used | 4,000 |
| Training images | 3,200 |
| Testing images | 800 |
| Gender Test Accuracy | **86%** |
| Male F1-score | **0.88** |
| Female F1-score | **0.85** |
| Age MAE | **9.79 years** |

## Limitations
- Age estimation has an MAE of 9.79 years.
- Hair length is estimated using an image-processing heuristic rather than a dedicated labelled hair dataset.
- Performance can be affected by lighting, pose, hair colour and image quality.
- Only 4,000 images were used for training/testing.

## Future Improvements
- Train on more images
- Add data augmentation
- Use transfer learning for age prediction
- Train a dedicated hair-length classifier
- Improve face/hair segmentation
- Add prediction confidence
- Deploy using Hugging Face Spaces, Streamlit or Flask

## Project Structure
```text
Long-Hair-Identification/
│
├── long-hair-identification(1).ipynb
├── README.md
├── requirements.txt
├── report.pdf
└── images/
```

## How to Run
1. Open the notebook.
2. Set the UTKFace dataset path.
3. Run the preprocessing cells.
4. Train the age and gender models.
5. Run the hair detection and decision-rule cells.
6. Launch the Gradio GUI.
7. Upload an image and view the result.

## Author
**Mrinmayi Kunkaliencar**  
Data Science & Machine Learning Project


## Conclusion

This project extends the Car Colour Detection training project with a new age-gated gender
prediction feature, reusing the same CNN architecture, training approach, and evaluation style.
The gender model reached 86% test accuracy and the age model achieved a 9.79-year MAE on a
4,000-image UTKFace subset. The age-gated long-hair rule was verified against both hand-picked
logic test cases and real dataset examples, confirming it correctly overrides the gender model's
own prediction for ages 20-30 based on hair length, while leaving predictions unaffected outside
that range.
