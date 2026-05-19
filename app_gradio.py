import cv2
import torch
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import numpy as np
import gradio as gr

# Cấu hình thiết bị (trên HF Spaces thường chỉ có CPU nếu dùng bản free)
DEVICE = torch.device("cpu")

EMOTION_DICT = {
    0: "Surprise", 1: "Fear", 2: "Disgust",
    3: "Happiness", 4: "Sadness", 5: "Anger", 6: "Neutral"
}

# Tải mô hình toàn cục
model = resnet50(num_classes=7)
checkpoint = torch.load('best_fer_model.pth', map_location=DEVICE)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def process_frame(frame):
    if frame is None:
        return None
    
    # Gradio truyền ảnh dưới dạng mảng NumPy RGB
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Thu nhỏ ảnh để nhận diện khuôn mặt nhanh hơn
    small_frame = cv2.resize(frame_bgr, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Nhân đôi toạ độ do ảnh đã bị thu nhỏ 1/2
        x, y, w, h = x * 2, y * 2, w * 2, h * 2
        
        margin_y = int(h * 0.1)
        margin_x = int(w * 0.1)
        y1 = max(0, y - margin_y)
        y2 = min(frame_bgr.shape[0], y + h + margin_y)
        x1 = max(0, x - margin_x)
        x2 = min(frame_bgr.shape[1], x + w + margin_x)

        face_roi = frame_bgr[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            emotion_idx = predicted.item()
            emotion_label = EMOTION_DICT.get(emotion_idx, "Unknown")
            conf_score = confidence.item() * 100
            display_text = f"{emotion_label} ({conf_score:.1f}%)"
        
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame_bgr, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

with gr.Blocks() as demo:
    gr.Markdown("# 🎭 Nhận diện cảm xúc khuôn mặt")
    gr.Markdown("Vui lòng cấp quyền sử dụng camera và hệ thống sẽ phân tích cảm xúc của bạn.")
    
    with gr.Row():
        input_img = gr.Image(sources=["webcam"], streaming=True, label="Camera trực tiếp")
        output_img = gr.Image(label="Kết quả dự đoán")
    
    # stream_every=0.2 giúp giảm tải (chỉ gửi 5 frame 1 giây thay vì gửi liên tục) để tránh bị lag queue
    input_img.stream(process_frame, inputs=[input_img], outputs=[output_img], stream_every=0.2)

if __name__ == "__main__":
    demo.launch()
