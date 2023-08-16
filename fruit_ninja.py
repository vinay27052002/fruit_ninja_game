import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2 #used for image processing techniques,landmark pb2 helps to find coordinates of hand features 
import time
import random
 
#creating class instance for drawing_utils from mediapipe.solutions,, to draw points and connectors of arm 
#then creating class instance for hands to get hand data 

mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands

#score variable to track the score 
#x_enemy ,y_enemy used to generate random coordinates of the circle that are needed to found using hands 

score=0 
x_enemy=random.randint(50,600)
y_enemy=random.randint(50,400)
 
 
 
def enemy():
  # giving scope throughout the program 

  global score,x_enemy,y_enemy

  cv2.circle(image, (x_enemy,y_enemy), 25, (0, 200, 0), 5)#image,coordinate,radius,colour,size of pen used to draw circle
  #score=score+1

#this will take input from the webcam 
video = cv2.VideoCapture(0)


# min_detection_confidence is for sensitivity of feature detection
#min_teacking_confidence if for tracking the hand moments 
#these values are stored in hands variable 

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while video.isOpened(): 
        #video data stores in frame variabe by using .read() function

        _, frame = video.read()
        #opencv takes input in BGR format & mediapipe takes input in RGB format 
        #to convert this using .cvtColor method 
 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #flipping the image 
         
        image = cv2.flip(image, 1)

        #image height and image width are height and width of image,helps in drawing the circle 
        
        imageHeight, imageWidth, _ = image.shape

        #.process from mediapipe.solution.hands ,processes the input to find the hand features

        results = hands.process(image)
   

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
        font=cv2.FONT_HERSHEY_SIMPLEX
        color=(255,0,255)
        text=cv2.putText(image,"Score",(480,30),font,1,color,4,cv2.LINE_AA)
        text=cv2.putText(image,str(score),(590,30),font,1,color,4,cv2.LINE_AA)
 
        enemy()
 
        if results.multi_hand_landmarks:#checks the hand is in frame , if yes then 
            for num, hand in enumerate(results.multi_hand_landmarks):#iterating on the all landmarks
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
 
 
        if results.multi_hand_landmarks != None:
          for handLandmarks in results.multi_hand_landmarks:
            for point in mp_hands.HandLandmark:
 
    
                normalizedLandmark = handLandmarks.landmark[point]
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
    
                point=str(point) #after we got coordinate we convert it into string to check index finger tip 
                #print(point)
                if point=='HandLandmark.INDEX_FINGER_TIP':
                 try:
                     cv2.circle(image, (pixelCoordinatesLandmark[0], pixelCoordinatesLandmark[1]), 25, (0, 200, 0), 5) #after getting index finger tip we are drawing circle 
                     
                     if pixelCoordinatesLandmark[0]==x_enemy or pixelCoordinatesLandmark[0]==x_enemy+10 or pixelCoordinatesLandmark[0]==x_enemy-10:
                        #if pixelCoordinatesLandmark[1]==y_enemy or pixelCoordinatesLandmark[1]==y_enemy+10 or pixelCoordinatesLandmark[1]==y_enemy-10:
                      #if pixelCoordinatesLandmark[1]==y_enemy or pixelCoordinatesLandmark[1]==y_enemy+10 or pixelCoordinatesLandmark[1]==y_enemy-10:
                        print("found")
                        x_enemy=random.randint(50,600)
                        y_enemy=random.randint(50,400)
                        score=score+1
                        font=cv2.FONT_HERSHEY_SIMPLEX
                        color=(255,0,255)
                        text=cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)
                        enemy()
                 except:
                  pass
        
        cv2.imshow('Hand Tracking', image) #displaying the image 
        #time.sleep(1)
 
        if cv2.waitKey(10) & 0xFF == ord('q'): # if q is pressed then the game quits by printing the score
            print("your score is",score)
            break
 
video.release()
cv2.destroyAllWindows()