# Age and Emotion Detection through Voice

## Project Overview
This project implements an intelligent voice analysis system using Deep Learning and Audio Signal Processing.
The system uses three Convolutional Neural Networks (CNNs), trained on mel-spectrogram images generated from real voice recordings, to detect a speaker's gender, age, and (for senior citizens) emotion.
It can:
- Detect whether a voice is male or female
- Reject female voices with the message "Upload male voice."
- Predict the speaker's age
- Mark speakers over 60 as senior citizens
- Detect emotion, but only for senior citizens
- Display results through a Gradio GUI

## Problem Statement
The objective of this project is to develop a machine learning system capable of:
- Detecting the gender of a speaker from a voice note
- Rejecting non-male voices with an appropriate message
- Predicting the age of male speakers
- Identifying senior citizens (age > 60)
- Detecting emotion for senior citizens only
- Displaying the results using a graphical user interface (GUI)

## Dataset
### CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
Speech dataset containing voice recordings from 91 actors with real documented ages (20-74 years), fetched live from the dataset's official demographics file. Used for gender, age, and emotion training.
Emotion classes include:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Additional speech dataset used to add more gender and emotion training examples. Actors are all young/mid-life, so this dataset is excluded from age training.

## Technologies Used
- Python
- TensorFlow / Keras
- Librosa
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Gradio

## Methodology
### Step 1
Load each voice clip and convert it into a mel-spectrogram image using Librosa.

### Step 2
Resize each spectrogram to 128x128 with OpenCV, the same way images were resized in the car colour project.

### Step 3
Train a CNN model to classify gender (male/female).

### Step 4
Train a CNN model to estimate age, using only male voices with known real ages.

### Step 5
Train a CNN model to classify emotion (angry, disgust, fear, happy, neutral, sad).

### Step 6
Chain the models together:
- Reject the voice if it is not male
- Estimate age if it is male
- Detect emotion only if the estimated age is above 60

### Step 7
Display:
- Gender
- Estimated Age
- Senior Citizen status
- Detected Emotion (senior citizens only)

## Results
The developed system successfully:
- Detects gender from voice
- Rejects female voices with "Upload male voice."
- Predicts age for male speakers
- Marks speakers over 60 as senior citizens
- Detects emotion for senior citizens

Gender CNN Validation Accuracy:
**~92%**

Age CNN Mean Absolute Error:
**13.8 years**

Senior Citizen (60+) Threshold Accuracy:
**90.3%**

Emotion CNN Validation Accuracy:
**~47-50%**

## Visual Outputs
The notebook includes:
- Training Accuracy Graph
- Validation Accuracy Graph
- Training Loss Graph
- Validation Loss Graph
- Confusion Matrix (Emotion)
- Sample Spectrogram Images
- GUI Output

## Project Structure
Age-and-Emotion-Detection-through-Voice/
│
├── Age_and_Emotion_Detection_through_Voice.ipynb
├── README.md
├── requirements.txt
├── gender_cnn.keras
├── age_cnn.keras
├── emotion_cnn.keras
└── test_clips/

## Installation
Install the required packages:
```bash
pip install tensorflow librosa soundfile opencv-python gradio matplotlib scikit-learn
```

## How to Run
1. Open the notebook.
2. Add the CREMA-D and RAVDESS datasets as input.
3. Train or load the CNN models (gender, age, emotion).
4. Run the spectrogram and prediction pipeline cells.
5. Launch the Gradio GUI.
6. Upload or record a voice note.
7. View the detection results.

## Known Limitations
- Age prediction accuracy is limited by the small number of senior actors (only 5 male actors over 60) in the training data.
- The emotion model is trained on acted, exaggerated speech and may score lower on natural conversation.
- No clinical audio tools (e.g. Praat) were used; all audio features come from Librosa.

## Future Improvements
- Real-time voice stream processing
- Larger senior-voice dataset for better age accuracy
- Multi-language support
- Flask/Streamlit deployment

## Author
**Mrinmayi Kunkaliencar**
Data Science & Machine Learning Project
