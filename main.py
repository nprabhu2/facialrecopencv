# Import all the necessary libraries
import pathlib
import cv2

# Initialize the Classifier with the path to the Haar Cascade Classifier
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

#Start the Live Camera Feed
camera = cv2.VideoCapture(0)

#Infinite loop that takes every frame, converts to grayscale for easy face detection.
while True:
    _, frame = camera.read()
    if frame is None: 
        print("Failed to capture frame. Camera may be inaccessible.")
        continue  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
#Puts box around subject when in frame and Displays on live feed.
    for (x, y, width, height) in faces:
        cv2.rectangle(frame,(x, y), (x+width, y+height), (255,255,0), 2)
    
    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break
#When program is closed Camera is turned off and OpenCV instances are closed.
camera.release()
cv2.destroyAllWindows()
