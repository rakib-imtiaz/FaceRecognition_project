from flask import Flask, render_template, Response, redirect, url_for, session, flash, jsonify, current_app
import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from functools import wraps
from werkzeug.local import LocalProxy

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()

# Create a global variable for authentication status
auth_status = {'success': False}

# Add this global variable at the top with other globals
signup_complete = {'status': False}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def face_extractor(img):
    if img is None:
        return None
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        return cropped_face
    except Exception as e:
        print(f"Error in face_extractor: {str(e)}")
        return None

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup')
def signup():
    signup_complete['status'] = False  # Reset status at signup page load
    return render_template('signup.html')

@app.route('/login')
def login():
    auth_status['success'] = False  # Reset status at login page load
    return render_template('login.html')

@app.route('/verify_login')
def verify_login():
    return Response(gen_frames_login(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_auth_status')
def check_auth_status():
    print("Checking auth status:", auth_status['success'])  # Debug print
    if auth_status['success']:
        # Reset the status after checking
        auth_status['success'] = False
        session['logged_in'] = True
        return jsonify({'status': 'success'})
    return jsonify({'status': 'pending'})

@app.route('/reset_auth')
def reset_auth():
    auth_status['success'] = False
    return jsonify({'status': 'reset'})

@app.route('/dashboard')
@login_required
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

def gen_frames_signup():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open camera")
        
        count = 0
        while count < 100:
            ret, frame = cap.read()
            if not ret:
                break
                
            face = face_extractor(frame)
            
            # Add guidance text
            if face is None:
                cv2.putText(frame, "Look straight at the camera", (50, 50), 
                           cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "No face detected!", (50, 100), 
                           cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            else:
                # Get face dimensions and position
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    # Draw face frame
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Calculate center of frame and face
                    frame_center_x = frame.shape[1] // 2
                    frame_center_y = frame.shape[0] // 2
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2
                    
                    # Define acceptable range for face position (center ± 20%)
                    x_tolerance = frame.shape[1] * 0.2
                    y_tolerance = frame.shape[0] * 0.2
                    
                    # Check face position and give guidance
                    if abs(face_center_x - frame_center_x) > x_tolerance:
                        if face_center_x < frame_center_x:
                            cv2.putText(frame, "Move right", (50, 50), 
                                      cv2.FONT_HERSHEY_COMPLEX, 1, (0, 165, 255), 2)
                        else:
                            cv2.putText(frame, "Move left", (50, 50), 
                                      cv2.FONT_HERSHEY_COMPLEX, 1, (0, 165, 255), 2)
                    
                    if abs(face_center_y - frame_center_y) > y_tolerance:
                        if face_center_y < frame_center_y:
                            cv2.putText(frame, "Move down", (50, 100), 
                                      cv2.FONT_HERSHEY_COMPLEX, 1, (0, 165, 255), 2)
                        else:
                            cv2.putText(frame, "Move up", (50, 100), 
                                      cv2.FONT_HERSHEY_COMPLEX, 1, (0, 165, 255), 2)
                    
                    # If face is well-positioned, take photo
                    if (abs(face_center_x - frame_center_x) <= x_tolerance and 
                        abs(face_center_y - frame_center_y) <= y_tolerance):
                        
                        count += 1
                        face = cv2.resize(face, (200, 200))
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        
                        if not os.path.isdir("Captured_Images"):
                            os.makedirs("Captured_Images")
                        
                        file_name_path = f'Captured_Images/{count}.jpg'
                        cv2.imwrite(file_name_path, face)
                        
                        # Show progress
                        cv2.putText(frame, f'Photo: {count}/100', (frame.shape[1]-200, 50), 
                                  cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, "Perfect Position!", (50, 50), 
                                  cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    
            # Add guide overlay
            height, width = frame.shape[:2]
            # Draw center target
            cv2.circle(frame, (width//2, height//2), 100, (0, 255, 0), 2)
            cv2.line(frame, (width//2-120, height//2), (width//2+120, height//2), (0, 255, 0), 1)
            cv2.line(frame, (width//2, height//2-120), (width//2, height//2+120), (0, 255, 0), 1)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        cap.release()
        if count >= 100:
            train_model()
            signup_complete['status'] = True
            
    except Exception as e:
        print(f"Camera error: {str(e)}")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera not available", (50, 240),
                   cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def train_model():
    data_path = os.getcwd() + '/Captured_Images/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)

    # Train only once
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    model.write('trainer.yml')
    print("Model trained successfully")

def gen_frames_login():
    cap = cv2.VideoCapture(0)
    # Load model once at the start
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read('trainer.yml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        image, face = face_detector(frame)
        
        try:
            if len(face) > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                # Use the already loaded model
                results = model.predict(face)
                
                if results[1] < 500:
                    confidence = int(100 * (1 - (results[1])/400))
                    display_string = str(confidence) + '% Confident it is User'
                    
                    cv2.putText(image, display_string, (100, 120), 
                              cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
                    
                    if confidence > 90:
                        cv2.putText(image, "Unlocked", (250, 450), 
                                  cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                        auth_status['success'] = True
                        ret, buffer = cv2.imencode('.jpg', image)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        cap.release()
                        break
                    else:
                        cv2.putText(image, "Locked", (250, 450), 
                                  cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                else:
                    cv2.putText(image, "Locked", (250, 450), 
                              cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                    
            else:
                cv2.putText(image, "No Face Found", (220, 120), 
                          cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                cv2.putText(image, "Locked", (250, 450), 
                          cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                
        except Exception as e:
            print(f"Error: {str(e)}")
            cv2.putText(image, "No Face Found", (220, 120), 
                      cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.putText(image, "Locked", (250, 450), 
                      cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed_signup')
def video_feed_signup():
    return Response(gen_frames_signup(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('index'))

@app.route('/check_signup_status')
def check_signup_status():
    if signup_complete['status']:
        signup_complete['status'] = False  # Reset the status
        return jsonify({'status': 'success'})
    return jsonify({'status': 'pending'})

if __name__ == '__main__':
    if os.path.exists('trained_model.yml'):
        model.read('trained_model.yml')
    app.run(debug=True) 