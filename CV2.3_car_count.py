import cv2
import numpy as np
from collections import OrderedDict
import math

# Class to track centroids of detected cars
# and manage their IDs
class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = np.linalg.norm(np.array(objectCentroids)[:, None] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                self.objects[objectIDs[row]] = input_centroids[col]
                self.disappeared[objectIDs[row]] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                self.disappeared[objectIDs[row]] += 1
                if self.disappeared[objectIDs[row]] > self.max_disappeared:
                    self.deregister(objectIDs[row])

            for col in unusedCols:
                self.register(input_centroids[col])

        return self.objects

# Main function to read video and track cars
cap = cv2.VideoCapture("road.mp4")
car_cascade = cv2.CascadeClassifier("cars.xml")

ct = CentroidTracker(max_disappeared=15)
total_cars = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 3, minSize=(60, 60))

    input_centroids = []
    for (x, y, w, h) in cars:
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        input_centroids.append((cx, cy))

    objects = ct.update(np.array(input_centroids))

    for (objectID, centroid) in objects.items():
        total_cars.add(objectID)
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    cv2.putText(frame, f"Total Cars: {len(total_cars)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Car Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()