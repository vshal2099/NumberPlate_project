import cv2
import os
import time

harcascade = "model/haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(harcascade)

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

os.makedirs("plates", exist_ok=True)
min_area = 500
count = 0
prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame.")
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 5)
    img_roi = None

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            img_roi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", img_roi)

    # FPS Display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Result", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and img_roi is not None:
        cv2.imwrite(f"plates/plate_{count}.jpg", img_roi)
        print(f"Plate saved: plates/plate_{count}.jpg")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
