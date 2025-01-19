from flask import Flask, render_template, Response
import cv2
import dlib
from math import hypot
import pyautogui
import mediapipe as mp
import threading

app = Flask(__name__)

class FacialAnalysisSystem:
    def __init__(self, predictor_path='shape_predictor_68_face_landmarks.dat'):
        # Initialize the face detector and landmark predictor
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(predictor_path)

    def calculate_blinking_ratio(self, eye_points, facial_landmarks):
        # Calculate the ratio for blinking detection
        left_eye_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_eye_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        top_center = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        bottom_center = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

        horizontal_line_length = hypot((left_eye_point[0] - right_eye_point[0]), (left_eye_point[1] - right_eye_point[1]))
        vertical_line_length = hypot((top_center[0] - bottom_center[0]), (top_center[1] - bottom_center[1]))

        ratio = horizontal_line_length / vertical_line_length
        return ratio

class GestureProcessingUnit:
    def __init__(self):
        # Initialize the hand gesture recognition module
        self.mp_hands_module = mp.solutions.hands
        self.hands_processor = self.mp_hands_module.Hands()

    def recognize_duck_gesture(self, hand_landmarks, img):
        # Recognize the duck gesture and display corresponding text
        thumb_tip_y = hand_landmarks.landmark[self.mp_hands_module.HandLandmark.THUMB_TIP].y
        middle_finger_tip_y = hand_landmarks.landmark[self.mp_hands_module.HandLandmark.MIDDLE_FINGER_TIP].y

        if thumb_tip_y < middle_finger_tip_y:
            cv2.putText(img, "Duck", (1050, 70), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 0, 255), 2)
            initiate_duck_action(img)

def midpoint(p1, p2):
    # Calculate the midpoint between two points
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def trigger_jump_action():
    # Simulate the key press and release for jumping action
    pyautogui.keyDown('space')
    pyautogui.keyUp('space')

def initiate_duck_action(img):
    # Simulate the key press and release for ducking action
    pyautogui.keyDown('down')
    pyautogui.keyUp('down')

def process_video_frames():
    # Initialize the facial analysis system and gesture processing unit
    facial_system = FacialAnalysisSystem()
    gesture_processor = GestureProcessingUnit()

    # Open the video capture device
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture video frame
        success, frame = video_capture.read()
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        detected_faces = facial_system.face_detector(grayscale_frame)

        for detected_face in detected_faces:
            # Extract facial landmarks
            facial_landmarks = facial_system.landmark_predictor(grayscale_frame, detected_face)

            # Calculate blinking ratio and perform jump action if the ratio is high
            left_eye_ratio = facial_system.calculate_blinking_ratio((36, 37, 38, 39, 40, 41), facial_landmarks)
            right_eye_ratio = facial_system.calculate_blinking_ratio((42, 43, 44, 45, 46, 47), facial_landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if blinking_ratio > 4.0:
                cv2.putText(frame, "Jump", (40, 70), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 2)
                trigger_jump_action()

        # Process hand landmarks for gesture recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_detection_results = gesture_processor.hands_processor.process(rgb_frame)

        if hand_detection_results.multi_hand_landmarks:
            for hand_landmarks in hand_detection_results.multi_hand_landmarks:
                # Recognize duck gesture and perform corresponding action
                gesture_processor.recognize_duck_gesture(hand_landmarks, frame)

        # Convert frame to JPEG format for streaming
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def render_index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/video_feed')
def provide_video_feed():
    # Provide the video feed for the web page
    return Response(process_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the Flask application in a separate thread
    threading.Thread(target=process_video_frames).start()
    app.run(debug=True)