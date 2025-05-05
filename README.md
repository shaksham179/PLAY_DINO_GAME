import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Setup
pyautogui.FAILSAFE = False
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

pressing = False
last_press_time = time.time()

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        # Finger tips
        index_tip = (int(lm[8].x * w), int(lm[8].y * h))
        thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))

        # Finger states
        fingers_down = [
            lm[8].y > lm[6].y,
            lm[12].y > lm[10].y,
            lm[16].y > lm[14].y,
            lm[20].y > lm[18].y
        ]
        is_fist = all(fingers_down)
        is_pinch = distance(index_tip, thumb_tip) < 40

        # If fist or pinch: continuously press space
        if is_fist or is_pinch:
            pressing = True
        else:
            pressing = False

        # Space press loop (fast but controlled)
        if pressing and (time.time() - last_press_time > 0.08):  # 12.5 CPS
            pyautogui.press('space')
            last_press_time = time.time()
            cv2.putText(frame, "AUTO JUMP", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    else:
        pressing = False

    cv2.imshow("Geometry Dash Jump - Hold to Auto Jump", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
