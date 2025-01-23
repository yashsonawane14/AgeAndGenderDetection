from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import numpy as np

app = Flask(__name__)

# Set path for uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define model mean values for normalization
MODEL_MEAN_VALUES = [78.4263377603, 87.7689143744, 114.896287379]

# Load models
faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"
ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"
genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Helper function to detect faces
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(h / 150)), 8)
    return frameOpencvDnn, faceBoxes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Read and process the image
        image = cv2.imread(filepath)
        resultImg, faceBoxes = highlightFace(faceNet, image)

        if not faceBoxes:
            return "No face detected"

        for faceBox in faceBoxes:
            face = image[max(0, faceBox[1]):min(faceBox[3], image.shape[0] - 1),
                         max(0, faceBox[0]):min(faceBox[2], image.shape[1] - 1)]

            # Gender detection
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            # Age detection
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

        return render_template('result.html', gender=gender, age=age, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
