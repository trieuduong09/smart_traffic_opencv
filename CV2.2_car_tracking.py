import cv2
import numpy as np

car_cascade = cv2.CascadeClassifier('cars.xml')

# Hàm tính centroid của bounding box
def get_centroid(x, y, w, h):
    return (int(x + w/2), int(y + h/2))

cap = cv2.VideoCapture('road.mp4')

if not cap.isOpened():
    print("Không mở được file video!")
    exit()

trackers = []
centroids = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.05, 2)

    new_centroids = []
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centroid = get_centroid(x, y, w, h)
        new_centroids.append(centroid)
        cv2.circle(frame, centroid, 2, (0, 0, 255), -1)

    # Đơn giản: chỉ vẽ centroid, chưa gán ID cho từng xe
    centroids = new_centroids

    cv2.imshow('Car Tracking', frame)
    if cv2.waitKey(30) & 0xFF == 27:  # Nhấn ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()