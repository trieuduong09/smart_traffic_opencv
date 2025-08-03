import cv2

image = cv2.imread('car3.jpg')
if image is None:
    print("Error: Could not read the image.")
    exit()

car_cascade = cv2.CascadeClassifier('cars.xml')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

# Nhận diện xe hơi
cars = car_cascade.detectMultiScale(gray, 1.05, 1)

# Vẽ hình chữ nhật và đánh số
for idx, (x, y, w, h) in enumerate(cars, 1):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Tính vị trí chính giữa hình chữ nhật
    center_x = x + w // 2
    center_y = y + h // 2
    # Đánh số thứ tự ở giữa hình chữ nhật
    cv2.putText(image, str(idx), (center_x - 10, center_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Car Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()