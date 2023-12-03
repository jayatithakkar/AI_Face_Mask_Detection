import cv2
import pandas as pd
import os

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
PADDING = 20
DATASET_FOLDER = "FINAL-DATASET-FULL-NEW"
DIRECTORIES = ["train", "test", "val"]

# Function Definitions
def load_model(modelFile, configFile):
    model = cv2.dnn.readNet(modelFile, configFile)
    enable_cuda_in_network(model)
    return model

def enable_cuda_in_network(net):
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frameOpencvDnn.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def predict_age_gender(faceNet, ageNet, genderNet, image_path):
    frame = cv2.imread(image_path)
    _, faceBoxes = highlightFace(faceNet, frame)

    if not faceBoxes:
        return None, None

    faceBox = faceBoxes[0]  # Assuming one face per image for simplicity
    face = frame[max(0, faceBox[1]-PADDING):min(faceBox[3]+PADDING, frame.shape[0]-1),
                 max(0, faceBox[0]-PADDING):min(faceBox[2]+PADDING, frame.shape[1]-1)]

    if face.size == 0:
        return None, None

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    genderNet.setInput(blob)
    gender = GENDER_LIST[genderNet.forward()[0].argmax()]

    ageNet.setInput(blob)
    age = AGE_LIST[ageNet.forward()[0].argmax()]

    return age[1:-1], gender

# Initialize models
faceNet = load_model("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
ageNet = load_model("age_net.caffemodel", "age_deploy.prototxt")
genderNet = load_model("gender_net.caffemodel", "gender_deploy.prototxt")

# Process each directory
for directory in DIRECTORIES:
    csv_file = os.path.join(DATASET_FOLDER, directory, f"{directory}_labels.csv")
    if not os.path.exists(csv_file):
        continue

    df = pd.read_csv(csv_file)
    df['Age'] = ''
    df['Gender'] = ''

    for index, row in df.iterrows():
        image_path = os.path.join(DATASET_FOLDER, directory, row['Label'], row['ImageName'])
        if not os.path.exists(image_path):
            continue

        age, gender = predict_age_gender(faceNet, ageNet, genderNet, image_path)
        if age and gender:
            df.at[index, 'Age'] = age
            df.at[index, 'Gender'] = gender

    new_csv_file = os.path.join(DATASET_FOLDER, directory, f"{directory}_updated_labels.csv")
    df.to_csv(new_csv_file, index=False)
