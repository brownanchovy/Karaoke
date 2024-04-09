import cv2
import mediapipe as mp
import pyautogui
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
second_y = 0
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x*frame_width)
                y = int(landmark.y*frame_height)
                if id == 20:
                    cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                    fifth_x = screen_width/frame_width*x
                    fifth_y = screen_height/frame_height*y
                if id == 16:
                    cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                    fourth_x = screen_width/frame_width*x
                    fourth_y = screen_height/frame_height*y
                if id == 12:
                    cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                    third_x = screen_width/frame_width*x
                    third_y = screen_height/frame_height*y
                if id == 8:
                    cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                    second_x = screen_width/frame_width*x
                    second_y = screen_height/frame_height*y
                if id == 4:
                    cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                    first_x = screen_width/frame_width*x
                    first_y = screen_height/frame_height*y

                    print('distance', abs(second_y - first_y))
                    if abs(second_y - first_y) < 50:
                        pyautogui.sleep(0.8)
                        if abs(second_y - first_y) < 50:
                            pyautogui.doubleClick() #doubleclick 구현
                        else:
                            pyautogui.click() #click 구현
                        pyautogui.sleep(1)
                    elif abs(second_y - first_y) < 200:
                        pyautogui.moveTo(second_x, second_y)
                    #start from here
                    '''
                    if abs(third_y - second_y) < 30 and :
                        pyautogui.s
                    '''
    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)

