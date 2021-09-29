# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 00:53:22 2021

@author: Terrick Boyd
Implemented dnn (DeepNeuralNetwork) classifier which is apart of the OpenCV libraries and this dnn classifier has a multibox detector
This can identify faces in poor light and from side profile or other unusual angles.
The dnn classifier uses a state of the art network called resonate (a.k.a. residual neural network )
"""


import cv2
import numpy as np
import imutils
import mediapipe as mp


#!Need Imutils (pip install imutils) to help with image processing!


# Caffe file is not shipped with opencv-python. CLone file from Github in the Demo2 Folder
# Caffe model and prototxt file paths
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "configfile.txt"

# Load dnn classifier

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
video_capture = cv2.VideoCapture(0)


# Load mediapipe drawing tools and holistic model for hands face and eye and full body tracking
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        frame = imutils.resize(frame, width=750)

        # Make Detections
        results = holistic.process(frame)

        # print(results.face_landmarks)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 1. Draw face landmarks
        # mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        #                           mp_drawing.DrawingSpec(
        #                               color=(80, 110, 10), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(
        #                               color=(80, 256, 121), thickness=1, circle_radius=1)
        #                           )

        # 2. Right hand
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )

        # 3. Left Hand
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            if confidence < 0.5:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()
video_capture.release()
