#face detection in still images
import cv2
from google.colab.patches import cv2_imshow
face_cascade = cv2.CascadeClassifier("/content/opencvTutorial/files/haarcascade_frontalface_default.xml")

img = cv2.imread("/content/anuj.jpg")
#converting  it to grascale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.5,5)
#print faces
for (x,y,w,h) in faces :
  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
cv2_imshow(img)
