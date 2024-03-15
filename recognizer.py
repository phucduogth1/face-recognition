import os
import cv2
import numpy as np
from keras.models import model_from_json
import keras.utils as image

# load model
model = model_from_json(open("ckpt/model_json.json", "r").read())
# load weights
model.load_weights('ckpt/model.h5')

face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# detec face

cap = cv2.VideoCapture(0)   # bật webcam

while True:
    # captures frame and returns boolean value and captured image
    ret, test_img = cap.read()
    if not ret:
        continue
    faces_detected = face_haar_cascade.detectMultiScale(test_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)
        # cropping region of interest i.e. face area from  image
        roi_color = test_img[y:y+w, x:x+h]
        roi_color = cv2.resize(roi_color, (299, 299))
        img_pixels = image.img_to_array(roi_color)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])
        name = ['0', '1', '2','3','4','5','6', '7']
        print('RECOGNIZE>>>: ', predictions[0])
        predicted = name[max_index]

        cv2.putText(test_img, predicted, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
