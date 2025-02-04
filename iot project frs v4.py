import cv2
import time


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

start_time = 0
countdown_duration = 10  # in seconds

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    num_faces = len(faces)

    if num_faces == 0:
        if start_time == 0:
            start_time = time.time()
        else:
            elapsed_time = time.time() - start_time
            remaining_time = countdown_duration - int(elapsed_time)
            if remaining_time <= 0:
                cv2.putText(frame, "Online exam has been terminated", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Exam terminating in {remaining_time} seconds", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        start_time = 0

    if num_faces > 2:
        cv2.putText(frame, "More than 2 faces detected. Exam terminated.", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
