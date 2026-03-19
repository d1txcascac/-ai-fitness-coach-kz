import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time

# Конфигурация страницы
st.set_page_config(page_title="AI Fitness Coach KZ", layout="wide")
st.title("🏋️ AI Fitness Coach - Pose Analysis Demo")
st.markdown("**Real-time squat form correction using YOLOv11-pose**")

# Загрузка модели (кеширование для скорости)
@st.cache_resource
def load_model():
    return YOLO("yolo11n-pose.pt")

model = load_model()

# Инициализация состояния камеры
if "cam_running" not in st.session_state:
    st.session_state.cam_running = False

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Start Analysis", use_container_width=True):
        st.session_state.cam_running = True
    if st.button("🛑 Stop Analysis", use_container_width=True):
        st.session_state.cam_running = False

# Основной цикл обработки видео
if st.session_state.cam_running:
    st_frame = st.empty()
    cap = cv2.VideoCapture(0)
    
    st.info("Perform squats! Monitor your knee angle (>90° for safety).")
    
    while cap.isOpened() and st.session_state.cam_running:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Анализ через YOLO11
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Логика анализа углов (пример для колена)
        if results[0].keypoints is not None:
            # Получение координат ключевых точек (17 точек)
            keypoints = results[0].keypoints.xyn[0].tolist()
            if len(keypoints) > 14:
                # Точки для колена (hip, knee, ankle)
                # Упрощенный вывод фидбека
                st.sidebar.success("Pose Detected: Analyzing form...")
        
        # Отображение кадра в Streamlit
        st_frame.image(annotated_frame, channels="BGR")
        time.sleep(0.01)
        
    cap.release()
else:
    st.write("Camera is off. Click 'Start Analysis' to begin.")
