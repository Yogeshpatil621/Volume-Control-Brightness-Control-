import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)
Draw = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin, volMax = volume.GetVolumeRange()[:2]

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Process = hands.process(frameRGB)
    landmarkList = []

    if Process.multi_hand_landmarks:
        for hand_index, handlm in enumerate(Process.multi_hand_landmarks):
            lm_list = []
            for _id, lm in enumerate(handlm.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([_id, cx, cy])
            landmarkList.append(lm_list)
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

        if len(landmarkList) > 0:
            # For volume control with the first hand (left hand)
            if len(landmarkList) > 0:
                x1, y1 = landmarkList[0][4][1], landmarkList[0][4][2]
                x2, y2 = landmarkList[0][8][1], landmarkList[0][8][2]
                cv2.circle(frame, (x1, y1), 7, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 7, (255, 0, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

                length = hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [15, 220], [volMin, volMax])
                volbar = np.interp(length, [15, 220], [400, 150])
                volper = np.interp(length, [15, 220], [0, 100])
                volume.SetMasterVolumeLevel(vol, None)

                cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 255), 4)
                cv2.rectangle(frame, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 0), 3)

            # For brightness control with the second hand (right hand)
            if len(landmarkList) > 1:
                x1, y1 = landmarkList[1][4][1], landmarkList[1][4][2]
                x2, y2 = landmarkList[1][8][1], landmarkList[1][8][2]
                cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                length = hypot(x2 - x1, y2 - y1)
                brightness = np.interp(length, [15, 220], [0, 100])
                sbc.set_brightness(int(brightness))
                
                cv2.putText(frame, f"Brightness: {int(brightness)}%", (450, 40), cv2.FONT_ITALIC, 1, (0, 255, 0), 3)

    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
