# Sign Language Detection

## Project Overview
This project implements an image-based and real-time American Sign Language (ASL) alphabet recognition system using Deep Learning and Computer Vision.
The system can:
Predict ASL alphabet letters from an uploaded image
Predict ASL alphabet letters from real-time webcam video
Combine predicted letters to recognize a set of known words
Restrict operation to a specified time window
Display results through a Gradio GUI

## Problem Statement
The objective is to develop a machine learning system capable of predicting ASL alphabet signs from images and video, and using the predicted letters for a task-specific known-word recognition feature.

### Decision Rule
For each predicted letter:
Letter matches `del` → remove the last character from the spelled word
Letter matches `space` → add a space to the spelled word
Letter matches `nothing` → ignored, no hand detected
Otherwise → letter is appended to the spelled word
If the spelled word matches an entry in the known-word list (HELLO, YES, NO, THANKS, PLEASE, LOVE), it is flagged as a recognized word.

## Dataset
ASL Alphabet Dataset (Kaggle)
The project uses the ASL Alphabet dataset. Images are organized into labeled folders by class.
Classes:
29 classes → A-Z, `del`, `nothing`, `space`

### Dataset Split
Total images: 87,000
Training images: 69,600
Validation images: 17,400

## Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
Gradio
openpyxl

## Methodology
### Step 1 — Data Preprocessing
Images are resized to 128 × 128 pixels and loaded directly from labeled folders. The data is split using an 80/20 train-validation split with a fixed random seed.

### Step 2 — Sign Classification CNN
A CNN is trained for 29-class sign classification.

Architecture:
Random Rotation / Zoom / Translation (augmentation)
Rescaling
Conv2D — 64 filters, BatchNorm, MaxPooling
Conv2D — 128 filters, BatchNorm, MaxPooling, Dropout — 0.25
Conv2D — 256 filters, BatchNorm, MaxPooling, Dropout — 0.25
Flatten
Dense — 256
Dropout — 0.4
Softmax output (29 classes)

Training:
Optimizer: Adam
Loss: Categorical Cross-Entropy
Epochs: 45 (early stopping patience 8, not triggered)
Batch size: 64
Learning rate scheduler: ReduceLROnPlateau (halved on plateau)

### Step 3 — Model Evaluation
The trained model is evaluated on the validation set using overall accuracy, a per-class classification report, and a confusion matrix.

### Step 4 — Known-Word Recognition
Predicted letters are combined, letter by letter, and checked against a fixed list of known words.

### Step 5 — Time-Restricted Operation
The predicted results are only served during a specified time window (6 PM–10 PM).

### Step 6 — Gradio GUI
The user uploads an image or streams from a webcam and receives:
Predicted Letter
Confidence
Word Spelled So Far
Status

## Results
| Metric | Result |
|---|---|
| Images used | 87,000 |
| Training images | 69,600 |
| Validation images | 17,400 |
| Validation Accuracy | 99.94% |
| Validation Loss | 0.0021 |
| Macro Avg F1-score | 1.00 |

## Limitations
Only recognizes static single-letter signs; does not recognize full ASL word-level gestures (e.g., signing "HELLO" as one motion) — words are only recognized by spelling them out letter by letter.
Real-world webcam accuracy can lag behind the reported validation accuracy, since the training images share a fairly consistent background/lighting per class.
J and Z are motion-based letters but are classified from single static frames.
Known-word recognition depends on every individual letter in the sequence being predicted correctly.

## Future Improvements
Train on a larger, more diverse dataset (varied backgrounds/lighting)
Add recognition of complete words and sentences as single gestures
Use transfer learning for faster convergence and improved generalization
Add confidence-based smoothing and prediction history to the GUI
Support dynamic signs and continuous gesture sequences
Deploy using Hugging Face Spaces, Streamlit, or Flask

## Conclusion
This project develops a CNN-based Sign Language Detection System for recognizing American Sign Language alphabet gestures from both static images and real-time webcam video. The model reached 99.94% validation accuracy on an 87,000-image ASL Alphabet subset, with near-perfect precision and recall across all 29 classes. Known-word recognition, time-restricted operation, and prediction logging were layered on top of the trained model and delivered through a single Gradio interface, confirming a practical, functioning end-to-end sign-language recognition application.

## Project Structure
Sign-Language-Detection/
│
├── sign-language-detection.ipynb
├── README.md
├── report.txt
└── images/

## How to Run
1. Open the notebook.
2. Set the ASL Alphabet dataset path (`DATA_DIR`).
3. Run the data-loading and preprocessing cells.
4. Train the CNN model.
5. Run the evaluation cells (accuracy, classification report, confusion matrix).
6. Launch the Gradio GUI.
7. Upload an image or use the webcam tab, and view the result.

## Author
**Mrinmayi Kunkaliencar**
Data Science & Machine Learning Project
