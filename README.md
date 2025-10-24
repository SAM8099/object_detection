# AI-Based People Counting System Using Dual-ROI Door Tracking and YOLOv8

## Tech Stack 
1. Python = 3.12
2. ultralytics
3. Opencv
4. YOLO V8 model
5. FastAPI

## File Structure
1. app.py : Contains the FastAPI backend for calling the functions. It takes input as video
2. test.py : Contains the function for counting people
3. research/experiments.ipynb : Experimented with the model to find the best approach
4. research/test2.mp4 : Sample video for testing
5. requirements.txt : Contains the frameworks and libraries required to run the app.

## How to use it?
1. pip install -r requirements.txt
2. uvicorn app:app --reload

