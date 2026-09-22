# Senior Citizen Identification System

## Project Overview
This project builds a video/webcam-based system that watches a mall or store entrance, detects every person in view, and predicts their age and gender using a Convolutional Neural Network trained from scratch. Anyone whose predicted age is above 60 is flagged as a senior citizen, and every visit is logged automatically.

The system can:
* Detect multiple people at once in a video or live webcam feed
* Predict each detected person's age
* Predict each detected person's gender
* Track each person across frames so a single visit isn't logged more than once
* Flag anyone with predicted age above 60 as a senior citizen
* Record the age, gender, and time of each visit to a CSV/Excel file
* Show live results through a Gradio GUI

## Problem Statement
The task is to develop a machine learning system that can predict multiple people's age and gender in a video or real-time webcam feed for a mall or local store setting, mark anyone whose predicted age exceeds 60 as a senior citizen along with their gender, and store the age, gender, and time of visit for every person in a CSV or Excel file.

## Dataset
**Age, Gender and Ethnicity (Face Data) CSV**

This dataset provides 48x48 grayscale face images with age and gender already available as columns, along with the pixel data itself, so no separate image folder or filename parsing is needed.

* `age` - the person's age (used as the regression target)
* `gender` - 0 = Male, 1 = Female
* `pixels` - the face image, stored as a space-separated string of pixel values

23,705 images are available in the full dataset, of which 2,395 (about 10%) are seniors (age > 60). 15,000 images were used for training and testing to keep training time reasonable, split 85/15 into 12,750 training images and 2,250 testing images.

## Technologies Used
* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Gradio

## Methodology
A single CNN is trained from scratch on the face dataset, with two output heads sharing one convolutional backbone: a sigmoid head for gender and a linear head for age. For video/webcam input, OpenCV's Haar cascade detects every face per frame, a lightweight overlap-based tracker follows each person across frames to avoid duplicate counting, and predictions are averaged over several frames before a visit is logged with its age, gender, and senior-citizen flag (age > 60) to `visit_log.csv` / `visit_log.xlsx`.

## Results
| Metric | Result |
|---|---:|
| Gender Test Accuracy | 85.2% |
| Gender F1-score | 0.847 |
| Age Test MAE | 7.44 years |
| Age R2 | 0.729 |
| Senior-Citizen Flag Accuracy | 93.7% |
| Senior-Citizen Flag Precision | 0.739 |
| Senior-Citizen Flag Recall | 0.526 |
| Senior-Citizen Flag F1-score | 0.614 |

The multi-person detection and tracking pipeline was also verified against real, unlabeled stock video footage of a mall walkway, confirming it correctly detects multiple visitors, avoids logging the same person repeatedly, and correctly flags the senior citizen present in the footage.

## Visual Outputs
The notebook includes:
* Age Distribution Histogram
* Gender Distribution Bar Chart
* Training Accuracy Graph
* Training Loss Graph
* Gender Confusion Matrix
* Sample Predictions on Test Faces
* Annotated Multi-Person Detection Output
* Gradio GUI Output

## Project Structure
```
Senior-Citizen-Identification/
│
├── seniorcitizenidentification.ipynb
├── README.md
├── requirements.txt
├── report.docx
└── outputs/
    ├── age_gender_cnn.keras
    ├── gender_confusion_matrix.png
    ├── training_metrics.txt
    ├── visit_log.csv
    └── visit_log.xlsx
```

## Installation
Install the required packages:
```
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn gradio openpyxl
```

## How to Run
1. Open the notebook.
2. Add the "Age, Gender and Ethnicity (Face Data) CSV" dataset as an input.
3. Run the preprocessing and training cells to build the age-gender CNN.
4. Run the face detection, tracking, and logging cells.
5. Launch the Gradio GUI.
6. Use the Live Webcam tab, or upload a video, to see detections.
7. Download the generated `visit_log.csv` / `visit_log.xlsx`.

## Future Improvements
- Train on the full dataset instead of a subset
- Replace the Haar cascade with a deep-learning face detector for crowded or poorly lit scenes
- Add motion prediction to the tracker for more robust occlusion handling
- Add confidence scores to each logged prediction
- Build a dashboard summarizing visitor statistics over time
- Deploy using Hugging Face Spaces, Streamlit, or Flask

## Author
**Mrinmayi Kunkaliencar**
Data Science & Machine Learning Project
