import cv2
import imutils
import mediapipe as mp
import threading

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cap = cv2.VideoCapture(camID)

    while True:
        ret, frame = cap.read()
        image = imutils.resize(frame, width=1000, height=1000)

        image = features(image)

        cv2.imshow(previewName, image)
        
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

# I tried to start threads in a loop to have an option to choose how many cameras we want to use at the moment
# However, when I applied it my laptop was about to explode

# number_of_cameras = int(intput("Enter number of cameras: "))
# for i in range(number_of_cameras):
#   t = camThread("Window", i)
#   t.start()

thread1 = camThread("Camera 1", 0)
# thread2 = camThread("Camera 2", 1)
# Uncomment when second camera is connected


thread1.start()
# thread2.start()


print()
print("Active threads", threading.activeCount())

def features(img):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        results1 = holistic.process(img)
        mp_drawing.draw_landmarks(
				img, results1.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

			# Right hand
        mp_drawing.draw_landmarks(
				img, results1.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

			# Left Hand
        mp_drawing.draw_landmarks(
				img, results1.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

			# Pose Detections
        mp_drawing.draw_landmarks(
				img, results1.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    return img