import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from datetime import datetime
from typing import Dict, Any, Tuple
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="AI Gym Posture Trainer",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .countdown-text {
        font-size: 8rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        font-size: 1.2rem;
        padding: 0.8rem;
        border: none;
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Core Analysis Classes
class PoseAnalyzer:
    @staticmethod
    def calculate_angle(point1: Tuple[float, float], point2: Tuple[float, float], point3: Tuple[float, float]) -> float:
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        radians = np.arccos(np.clip(np.dot(a-b, c-b) / (np.linalg.norm(a-b) * np.linalg.norm(c-b)), -1.0, 1.0))
        angle = np.degrees(radians)
        return angle
    
    @staticmethod
    def get_landmark_coordinates(landmarks, landmark_index: int) -> Tuple[float, float]:
        if landmarks and len(landmarks) > landmark_index:
            return (landmarks[landmark_index][0], landmarks[landmark_index][1])
        return (0, 0)

class ExerciseFormAnalyzer:
    def __init__(self):
        self.pose_analyzer = PoseAnalyzer()
    
    def analyze_squat(self, landmarks) -> Dict[str, Any]:
        if not landmarks:
            return {"valid": False, "feedback": "No pose detected"}
        
        left_hip = self.pose_analyzer.get_landmark_coordinates(landmarks, 23)
        left_knee = self.pose_analyzer.get_landmark_coordinates(landmarks, 25)
        left_ankle = self.pose_analyzer.get_landmark_coordinates(landmarks, 27)
        
        knee_angle = self.pose_analyzer.calculate_angle(left_hip, left_knee, left_ankle)
        
        if knee_angle > 160:
            phase = "standing"
            feedback = "Good starting position"
        elif knee_angle > 90:
            phase = "partial_squat"
            feedback = "Go deeper for full squat"
        else:
            phase = "deep_squat"
            feedback = "Excellent squat depth!"
        
        return {
            "valid": True,
            "exercise": "squat",
            "phase": phase,
            "knee_angle": knee_angle,
            "feedback": feedback,
            "form_score": min(100, max(0, 100 - abs(90 - knee_angle)))
        }
    
    def analyze_pushup(self, landmarks) -> Dict[str, Any]:
        if not landmarks:
            return {"valid": False, "feedback": "No pose detected"}
        
        left_shoulder = self.pose_analyzer.get_landmark_coordinates(landmarks, 11)
        left_elbow = self.pose_analyzer.get_landmark_coordinates(landmarks, 13)
        left_wrist = self.pose_analyzer.get_landmark_coordinates(landmarks, 15)
        
        elbow_angle = self.pose_analyzer.calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        if elbow_angle > 160:
            phase = "up_position"
            feedback = "Good extension"
        elif elbow_angle < 90:
            phase = "down_position"
            feedback = "Good depth - now push up!"
        else:
            phase = "mid_position"
            feedback = "Keep going"
        
        return {
            "valid": True,
            "exercise": "pushup",
            "phase": phase,
            "elbow_angle": elbow_angle,
            "feedback": feedback,
            "form_score": min(100, max(0, 100 - abs(elbow_angle - 90) / 2))
        }

class LiveVideoProcessor:
    def __init__(self, exercise_type="squat"):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.exercise_analyzer = ExerciseFormAnalyzer()
        self.exercise_type = exercise_type
        self.session_data = []
    
    def extract_landmarks(self, pose_landmarks, frame_shape):
        landmarks = []
        height, width, _ = frame_shape
        
        for landmark in pose_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append([x, y, landmark.z])
        
        return landmarks
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb_frame)
        
        exercise_analysis = {"valid": False}
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            landmarks = self.extract_landmarks(results.pose_landmarks, frame.shape)
            
            if self.exercise_type == "squat":
                exercise_analysis = self.exercise_analyzer.analyze_squat(landmarks)
            elif self.exercise_type == "pushup":
                exercise_analysis = self.exercise_analyzer.analyze_pushup(landmarks)
            
            frame = self.annotate_frame(frame, exercise_analysis)
            self.session_data.append(exercise_analysis)
        else:
            cv2.putText(frame, "No pose detected - Please position yourself in frame", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, exercise_analysis

    def annotate_frame(self, frame, exercise_analysis: Dict[str, Any]):
        form_score = exercise_analysis.get('form_score', 0)
        
        if form_score >= 80:
            color = (0, 255, 0)
        elif form_score >= 60:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        
        feedback = exercise_analysis.get('feedback', '')
        cv2.putText(frame, feedback, (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        score_text = f"Form Score: {form_score:.1f}/100"
        cv2.putText(frame, score_text, (30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        exercise = exercise_analysis.get('exercise', 'Unknown')
        phase = exercise_analysis.get('phase', 'Unknown')
        info_text = f"{exercise.title()}: {phase.replace('_', ' ').title()}"
        cv2.putText(frame, info_text, (30, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame

# Initialize session state
if 'workout_started' not in st.session_state:
    st.session_state.workout_started = False
if 'countdown_active' not in st.session_state:
    st.session_state.countdown_active = False
if 'session_data' not in st.session_state:
    st.session_state.session_data = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = None

# Header
st.markdown('<h1 class="main-header">💪 AI Gym Posture Trainer</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    exercise_type = st.selectbox(
        "Select Exercise",
        ["squat", "pushup"],
        help="Choose the exercise you want to perform"
    )
    
    st.markdown("---")
    st.subheader("📊 Session Stats")
    
    if st.session_state.session_data:
        avg_score = np.mean([d.get('form_score', 0) for d in st.session_state.session_data if d.get('valid', False)])
        st.metric("Average Form Score", f"{avg_score:.1f}/100")
        st.metric("Total Frames Analyzed", len(st.session_state.session_data))
        
        if st.session_state.start_time:
            duration = time.time() - st.session_state.start_time
            st.metric("Session Duration", f"{int(duration)}s")
    
    st.markdown("---")
    
    if st.button("🔄 Reset Session"):
        st.session_state.session_data = []
        st.session_state.workout_started = False
        st.session_state.countdown_active = False
        st.session_state.start_time = None
        st.rerun()
    
    if st.session_state.session_data and st.button("📥 Export Session Data"):
        df = pd.DataFrame(st.session_state.session_data)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"workout_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if not st.session_state.workout_started and not st.session_state.countdown_active:
        st.info("👋 Position yourself in front of the camera and click START when ready!")
        if st.button("🚀 START WORKOUT", key="start_btn"):
            st.session_state.countdown_active = True
            st.rerun()
    
    if st.session_state.countdown_active:
        countdown_placeholder = st.empty()
        
        for i in range(5, 0, -1):
            with countdown_placeholder.container():
                st.markdown(f'<div class="countdown-text">{i}</div>', unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; font-size: 1.5rem;'>Get Ready! 💪</p>", unsafe_allow_html=True)
            time.sleep(1)
        
        countdown_placeholder.empty()
        st.session_state.countdown_active = False
        st.session_state.workout_started = True
        st.session_state.start_time = time.time()
        st.rerun()

# Video feed
if st.session_state.workout_started:
    st.markdown("### 📹 Live Video Feed")
    
    video_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    col_stop1, col_stop2, col_stop3 = st.columns([1, 1, 1])
    with col_stop2:
        stop_button = st.button("⏹️ STOP WORKOUT", key="stop_btn")
    
    if stop_button:
        st.session_state.workout_started = False
        st.success("✅ Workout session completed!")
        st.balloons()
        st.rerun()
    
    # Initialize video processor
    processor = LiveVideoProcessor(exercise_type=exercise_type)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check your camera permissions.")
    else:
        frame_count = 0
        
        while st.session_state.workout_started:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Failed to capture frame")
                break
            
            # Process frame
            processed_frame, analysis = processor.process_frame(frame)
            
            # Store session data
            if analysis.get('valid', False):
                st.session_state.session_data.append(analysis)
            
            # Convert BGR to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
            
            # Display stats
            if analysis.get('valid', False):
                with stats_placeholder.container():
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Form Score", f"{analysis.get('form_score', 0):.1f}")
                    with metric_cols[1]:
                        st.metric("Exercise", analysis.get('exercise', 'N/A').title())
                    with metric_cols[2]:
                        st.metric("Phase", analysis.get('phase', 'N/A').replace('_', ' ').title())
                    with metric_cols[3]:
                        angle_key = 'knee_angle' if exercise_type == 'squat' else 'elbow_angle'
                        angle_val = analysis.get(angle_key, 0)
                        st.metric("Angle", f"{angle_val:.1f}°")
            
            frame_count += 1
            
            # Break if stop button clicked
            if not st.session_state.workout_started:
                break
            
            # Small delay to control frame rate
            time.sleep(0.03)
        
        cap.release()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>Made with ❤️ using Streamlit, MediaPipe & OpenCV</p>",
    unsafe_allow_html=True
)