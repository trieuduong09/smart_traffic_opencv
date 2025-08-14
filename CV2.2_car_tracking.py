import cv2
import numpy as np
import os

cascade_path = 'cars.xml'
video_path = 'road.mp4'

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1]) if imageA.shape[0] > 0 and imageA.shape[1] > 0 else 1.0
    return err

def diffUpDown(img):
    height, width = img.shape[:2]
    if height < 2 or width < 1: return float('inf')
    half = height // 2
    top = img[:half, :width]
    bottom = img[half:, :width]
    if top.shape[0] == 0 or bottom.shape[0] == 0: return float('inf')
    top = cv2.flip(top, 1)
    try:
        bottom = cv2.resize(bottom, (32, 64), interpolation=cv2.INTER_LINEAR)
        top = cv2.resize(top, (32, 64), interpolation=cv2.INTER_LINEAR)
    except Exception:
         return float('inf')
    return mse(top, bottom)

def diffLeftRight(img):
    height, width = img.shape[:2]
    if height < 1 or width < 2: return float('inf')
    half = width // 2
    left = img[:height, :half]
    right = img[:height, half:]
    if left.shape[1] == 0 or right.shape[1] == 0: return float('inf')
    right = cv2.flip(right, 1)
    try:
        left = cv2.resize(left, (32, 64), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, (32, 64), interpolation=cv2.INTER_LINEAR)
    except Exception:
        return float('inf')
    return mse(left, right)

def isNewRoi(rx, ry, rw, rh, rectangles):
    for r in rectangles:
        if len(r) == 4 and abs(r[0] - rx) < 40 and abs(r[1] - ry) < 40:
            return False
    return True

def detectRegionsOfInterest(frame, cascade):
    scaleDown = 2
    frameHeight, frameWidth = frame.shape[:2]
    if frameHeight <= 0 or frameWidth <= 0:
        return []

    try:
        frame_resized = cv2.resize(frame, (int(frameWidth / scaleDown), int(frameHeight / scaleDown)), interpolation=cv2.INTER_LINEAR)
    except Exception:
         return []

    frameHeight, frameWidth = frame_resized.shape[:2]
    if frameHeight <= 0 or frameWidth <= 0:
        return []

    cars = cascade.detectMultiScale(frame_resized, 1.2, 1)

    newRegions = []
    miny = int(frameHeight * 0.3)

    for (x, y, w, h) in cars:
        if w <= 0 or h <= 0: continue
        if x < 0: x = 0
        if y < 0: y = 0
        if x + w > frameWidth: w = frameWidth - x
        if y + h > frameHeight: h = frameHeight - y
        if w <= 0 or h <= 0: continue

        roiImage = frame_resized[y:y + h, x:x + w]

        if roiImage.size == 0: continue

        if y > miny:
            try:
                diffX = diffLeftRight(roiImage)
                diffY = round(diffUpDown(roiImage))

                if 1600 < diffX < 3000 and diffY > 12000:
                    rx, ry, rw, rh = x, y, w, h
                    newRegions.append([rx * scaleDown, ry * scaleDown, rw * scaleDown, rh * scaleDown])
            except Exception:
                 pass

    return newRegions

def detectCars(filename, cascade_path):
    if not os.path.exists(filename):
        print(f"Error: Video file not found at {filename}")
        return
    if not os.path.exists(cascade_path):
        print(f"Error: Cascade file not found at {cascade_path}")
        return

    rectangles = []
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
         print(f"Error loading cascade file: {cascade_path}")
         return

    vc = cv2.VideoCapture(filename)
    if not vc.isOpened():
         print(f"Error opening video file: {filename}")
         return

    frameCount = 0
    rval, frame = vc.read()

    while rval:
        if frame is None:
            rval, frame = vc.read()
            continue

        newRegions = detectRegionsOfInterest(frame.copy(), cascade) # Use copy to avoid modifying original frame unintentionally

        temp_rectangles = []
        for region in newRegions:
            if len(region) == 4:
                rx, ry, rw, rh = region
                if isNewRoi(rx, ry, rw, rh, rectangles):
                   temp_rectangles.append(region)

        rectangles.extend(temp_rectangles)

        draw_frame = frame.copy() # Draw on a copy
        for r in rectangles:
             if len(r) == 4:
                  cv2.rectangle(draw_frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 0, 255), 2)

        frameCount += 1
        if frameCount > 30:
            frameCount = 0
            rectangles = []

        cv2.imshow("Result", draw_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC key
            break

        rval, frame = vc.read()

    vc.release()
    cv2.destroyAllWindows()

detectCars(video_path, cascade_path)