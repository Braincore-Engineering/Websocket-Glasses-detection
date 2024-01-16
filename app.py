from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('response', {'data': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('request_prediction')
def handle_request_prediction():
    while True:
        ret, frame = camera.read()
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image_input = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image_input = (image_input / 127.5) - 1
        prediction = model.predict(image_input)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        # text = f"Class: {class_name[2:]}%"
        # cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        emit('prediction', {'image': img_bytes, 'class_name': class_name[2:], 'confidence': str(np.round(confidence_score * 100))[:-2]})

@socketio.on('disconnect_request')
def handle_disconnect_request():
    emit('disconnect', {'data': 'Disconnecting'})
    socketio.disconnect()

if __name__ == '__main__':
    socketio.run(app, debug=True, port = 5000)
