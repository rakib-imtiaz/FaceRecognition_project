# Face Recognition System

A Python-based face recognition system using OpenCV that captures face data and performs real-time face recognition.

## Prerequisites

- Python 3.x
- Webcam
- Git (optional)

## Installation and Setup

1. **Clone the repository** (if using Git)
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create and activate virtual environment**

   For Unix/macOS:
   ```bash
   python3 -m venv face_detection
   source face_detection/bin/activate
   ```

   For Windows:
   ```bash
   python3 -m venv face_detection
   face_detection\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Capture Training Data**
   - Run `faceRecognition.py`
   ```bash
   python faceRecognition.py
   ```
   - The system will capture 100 images of your face for training
   - Press 'Enter' to stop capturing before 100 images
   - Images will be saved in the `Captured_Images` directory

2. **Face Recognition**
   - After training, the system will automatically start face recognition
   - The system will display:
     - Confidence level of recognition
     - "Unlocked" status if confidence > 75%
     - "Locked" status if confidence < 75%
   - Press 'Enter' to exit the program

## Project Structure 