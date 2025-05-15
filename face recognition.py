#opencv library -> used for image processing, computer vision, and real-time applications
import cv2

#CascadeClassifier -> loads a pre-trained model for detecting faces
#cv2.data.haarcascades -> provides the path to the directory where Haar cascades are stored
#haarcascade_frontalface_default.xml is a file containing data for detecting front-facing human faces
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#accesses the default webcam (0) if there r more it'd be 1, 2 etc
video_capture = cv2.VideoCapture(0)

#function definition ->  takes a video frame as input and detects faces in it
def detect_bounding_box(vid):
    #converts the colour frame (vid) 2 grayscale -> req by the Haar cascade classifier
    #colour is unnecessary for face detection n grayscale is faster 2 process
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    #detects faces in the grayscale image
    #1.1 -> scaling factor
    #5 -> how many rectangles need to be detected near the current one for it to be retained (neighbours)
    #minSize -> filters out objects smaller than 40x40 px
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    #draws a cyan rectangle around each detected face
    for (x, y, w, h) in faces:
        #xy -> top left
        #x+w,y+h -> bottom right
        #4 -> border thickness
        cv2.rectangle(vid, (x, y), (x + w, y + h), (255, 255, 0), 4)
    #returns list of detected faces
    return faces


while True:
    #read frames from the video
    #video_frame -> contains actual video data
    result, video_frame = video_capture.read()
    #terminate the loop if the frame is not read successfully
    if result is False:
        break

    #calls the detect_bounding_box function to detect and mark faces on the frame
    faces = detect_bounding_box(
        video_frame
    )

    #display the processed frame in a window named as that
    cv2.imshow(
        "Face Detection", video_frame
    )

    #cv2.waitKey(1) -> waits 1 ms for a key press
    #if q key is pressed the prgm ends
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#releases the webcam for use by other apps
video_capture.release()
#closes all opencv windows opened during the program
cv2.destroyAllWindows()
