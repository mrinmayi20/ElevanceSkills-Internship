# Car Colour Detection and Traffic Analysis System

## Project Overview
This project implements an intelligent traffic analysis system using Deep Learning and Computer Vision.
The system combines a Convolutional Neural Network (CNN) for vehicle colour classification with YOLOv8 for object detection.
It can:
- Detect vehicles in traffic images
- Detect pedestrians
- Predict vehicle colours
- Count vehicles and pedestrians
- Highlight blue vehicles with red bounding boxes
- Highlight other vehicles with blue bounding boxes
- Display results through a Gradio GUI

## Problem Statement
The objective of this project is to develop a machine learning system capable of:
- Detecting vehicles and pedestrians in traffic images
- Predicting the colour of detected vehicles
- Counting vehicles and pedestrians
- Displaying the results using a graphical user interface (GUI)

## Dataset
### Vehicle Colour Recognition Dataset (VCoR)
Vehicle colour classification dataset containing images of vehicles with different colour classes.
Classes include:
- Beige
- Black
- Blue
- Brown
- Gold
- Green
- Grey
- Orange
- Pink
- Purple
- Red
- Silver
- Tan
- White
- Yellow

Traffic images are used for YOLOv8 object detection.

## Technologies Used
- Python
- TensorFlow / Keras
- YOLOv8
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Gradio

## Methodology
### Step 1
Train a CNN model to classify vehicle colours.

### Step 2
Use YOLOv8 to detect:
- Cars
- Trucks
- Buses
- Motorcycles
- Pedestrians

### Step 3
Crop each detected vehicle.

### Step 4
Predict the vehicle colour using the CNN.

### Step 5
Display:
- 🔴 Red bounding boxes for blue vehicles
- 🔵 Blue bounding boxes for other vehicles

### Step 6
Display:
- Total vehicles
- Total pedestrians
- 
## Results
The developed system successfully:
- Detects vehicles
- Detects pedestrians
- Predicts vehicle colours
- Counts vehicles
- Counts pedestrians
- Highlights blue vehicles with red boxes
- Highlights other vehicles with blue boxes

CNN Test Accuracy:
**78.86%**

## Visual Outputs
The notebook includes:
- Training Accuracy Graph
- Validation Accuracy Graph
- Training Loss Graph
- Validation Loss Graph
- Confusion Matrix
- Sample Predictions
- YOLO Detection Results
- GUI Output

## Project Structure
Car-Colour-Detection/
│
├── Car_Colour_Detection.ipynb
├── README.md
├── requirements.txt
├── report.pdf
└── images/

## Installation
Install the required packages:
```bash
pip install tensorflow ultralytics opencv-python gradio matplotlib scikit-learn

## How to Run
1. Open the notebook.
2. Train or load the CNN model.
3. Run the YOLO detection cells.
4. Launch the Gradio GUI.
5. Upload a traffic image.
6. View the detection results.

## Future Improvements
- Real-time video processing
- Vehicle tracking
- Traffic density estimation
- Multi-camera support
- Flask/Streamlit deployment

## Author
**Mrinmayi Kunkaliencar**
Data Science & Machine Learning Project
