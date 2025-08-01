import cv2

image = cv2.imread('car3.jpg') 

while image is None: print("Error: Could not read the image.")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Chuyển đổi ảnh sang thang độ xám
    
    cv2.imshow('Original Image', image) 
    cv2.waitKey(1600) # Thời gian hiển thị hình ảnh gốc 1.6 giây
    
    cv2.imshow('Grayscale Image', gray_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()