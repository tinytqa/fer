import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import numpy as np

# Cấu hình thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Từ điển ánh xạ nhãn
EMOTION_DICT = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral"
}

def load_model(weights_path):
    # Khởi tạo mô hình resnet50 với 7 classes
    # Do trong quá trình huấn luyện ResNet50 có thể lấy các weights từ torchvision,
    # nhưng ở đây ta chỉ cần cấu trúc rồi nạp weights đã lưu vào
    model = resnet50(num_classes=7)
    
    # Tải trọng số
    print(f"Đang tải trọng số từ {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    
    # Đôi khi checkpoint lưu toàn bộ dict hoặc chỉ lưu model.state_dict()
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    return model

# Khởi tạo biến transform (chuẩn hóa giống ImageNet do dùng ResNet50)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    print(f"Sử dụng thiết bị: {DEVICE}")
    try:
        model = load_model('best_fer_model.pth')
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}")
        return
    print("Tải mô hình thành công!")

    # Khởi tạo bộ nhận diện khuôn mặt của OpenCV
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không thể mở webcam. Vui lòng kiểm tra lại thiết bị.")
        return

    print("=========================================")
    print("Ứng dụng Nhận diện Cảm xúc đã sẵn sàng!")
    print("Nhấn 'q' để thoát.")
    print("=========================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ webcam.")
            break

        # Lật frame (mirror effect) để dễ nhìn hơn
        frame = cv2.flip(frame, 1)

        # Chuyển frame sang ảnh xám để nhận diện khuôn mặt (chính xác và nhanh hơn)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện các khuôn mặt trong frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Lấy vùng khuôn mặt (Region of Interest - ROI)
            # Dùng kích thước lớn hơn một chút (margin) nếu muốn bao phủ cả đầu
            margin_y = int(h * 0.1)
            margin_x = int(w * 0.1)
            
            y1 = max(0, y - margin_y)
            y2 = min(frame.shape[0], y + h + margin_y)
            x1 = max(0, x - margin_x)
            x2 = min(frame.shape[1], x + w + margin_x)

            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                continue

            # Chuyển đổi BGR (OpenCV) sang RGB (PIL/PyTorch)
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            
            # Áp dụng transforms
            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
            
            # Dự đoán
            with torch.no_grad():
                outputs = model(input_tensor)
                # Tính xác suất bằng hàm softmax
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion_idx = predicted.item()
                emotion_label = EMOTION_DICT.get(emotion_idx, "Unknown")
                conf_score = confidence.item() * 100
                
                # Tạo chuỗi hiển thị gồm Tên cảm xúc và Phần trăm
                display_text = f"{emotion_label} ({conf_score:.1f}%)"
            
            # Vẽ hình chữ nhật bao quanh khuôn mặt
            color = (0, 255, 0) # Xanh lá
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Viết label cảm xúc phía trên hộp khuôn mặt
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Hiển thị frame
        cv2.imshow('Facial Emotion Recognition', frame)

        # Thoát nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
