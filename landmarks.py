import mediapipe as mp 
import cv2 as cv
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands 
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

ptime = 0
ctime = 0


while True:
    is_True, frame = cap.read()

    imagergb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(imagergb)


    if results.multi_hand_landmarks:

    	for handlms in results.multi_hand_landmarks:
    		for lm_id, lm in enumerate(handlms.landmark):
    			# print(lm_id, lm)
    			h, w, c = frame.shape
    			cx, cy = int(lm.x*w),int(lm.y*h)
    			if lm_id == 17:
    			    cv.circle(frame, (cx, cy), 15, (255,255,0), cv.FILLED)

    		mpdraw.draw_landmarks(frame, handlms, mpHands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv.putText(frame, f"FPS: {int(fps)}", (10, 70), cv.FONT_HERSHEY_PLAIN,3,(0, 255, 0), 3)

    
    cv.imshow('frame',frame)

    cv.waitKey(1)

