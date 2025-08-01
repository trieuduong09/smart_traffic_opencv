import cv2

image = cv2.imread('car3.jpg')

while image is None: print("Error: Could not read the image.")
else:
    # vẽ hình chữ nhật màu vàng vào giữa ảnh
    height, width, _ = image.shape
    
    # tính toán tọa độ của hình chữ nhật
    start_point = (width // 4, height // 4)
    end_point = (3 * width // 4, 3 * height // 4)
    
    color = (0, 255, 255)
    thickness = 2  # độ dày của đường viền
    
    cv2.rectangle(image, start_point, end_point, color, thickness) 
    
    # hiển thị tên lên khung chữ nhật
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    text = "Nguyen Trieu Duong"
    
    # lấy kích thước của văn bản cỡ chữ 1 độ dày 2 
    text_size = cv2.getTextSize(text, font, 1, 2)[0] 
    
    # tính toán vị trí để căn giữa văn bản
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(image, text, (text_x, text_y), font, 1, (0, 0, 255), 2)
    
    cv2.imshow('Image with Rectangle and Name', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
