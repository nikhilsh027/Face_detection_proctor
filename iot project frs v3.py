import cv2
import time

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Flag to track if a face is recognized
face_detected = False

# Timer variables
start_time = 0
countdown_duration = 10  # in seconds

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        face_detected = True
        start_time = 0  # reset the timer if face is detected
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    else:
        if face_detected:
            face_detected = False  # reset the flag if face was detected before
            start_time = time.time()  # start the timer if no face is detected
        else:
            if start_time != 0:
                elapsed_time = time.time() - start_time
                remaining_time = countdown_duration - int(elapsed_time)
                if remaining_time > 0:
                    # Display countdown message
                    cv2.putText(frame, f"Exam terminating in {remaining_time} seconds", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Terminate exam after countdown finishes
                    cv2.putText(frame, "Online exam has been terminated", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # You can add additional logic here to perform any necessary actions upon exam termination
                    # For example, save progress, close application, etc.

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
