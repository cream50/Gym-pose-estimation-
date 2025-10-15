import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from datetime import datetime
from typing import Dict, Any, Tuple, List

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Gym Posture Trainer",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
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
    .report-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4ECDC4;
    }
    .excellent-badge {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .good-badge {
        background: linear-gradient(90deg, #f2994a 0%, #f2c94c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .needs-improvement-badge {
        background: linear-gradient(90deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# CORE CLASSES
# ---------------------------------------------------
class PoseAnalyzer:
    @staticmethod
    def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        # Handle potential zero vector or float errors
        norm_a_b = np.linalg.norm(a-b)
        norm_c_b = np.linalg.norm(c-b)
        if norm_a_b == 0 or norm_c_b == 0:
            return 0.0
        dot_product = np.dot(a-b, c-b)
        cosine_angle = np.clip(dot_product / (norm_a_b * norm_c_b), -1.0, 1.0)
        radians = np.arccos(cosine_angle)
        return np.degrees(radians)
    
    @staticmethod
    def get_landmark_coordinates(landmarks, index: int) -> Tuple[float, float]:
        if landmarks and len(landmarks) > index:
            # Note: Landmarks here are expected to be the unscaled (0-1) coordinates
            return (landmarks[index][0], landmarks[index][1])
        return (0, 0)

class ExerciseFormAnalyzer:
    def __init__(self):
        self.pose_analyzer = PoseAnalyzer()
    
    def analyze_squat(self, landmarks):
        if not landmarks:
            return {"valid": False, "feedback": "No pose detected"}
        hip = self.pose_analyzer.get_landmark_coordinates(landmarks, 23)
        knee = self.pose_analyzer.get_landmark_coordinates(landmarks, 25)
        ankle = self.pose_analyzer.get_landmark_coordinates(landmarks, 27)
        knee_angle = self.pose_analyzer.calculate_angle(hip, knee, ankle)
        if knee_angle > 160:
            phase, feedback = "standing", "Good starting position"
        elif knee_angle > 90:
            phase, feedback = "partial_squat", "Go deeper for full squat"
        else:
            phase, feedback = "deep_squat", "Excellent squat depth!"
        return {
            "valid": True, "exercise": "squat", "phase": phase,
            "knee_angle": knee_angle, "feedback": feedback,
            "form_score": min(100, max(0, 100 - abs(90 - knee_angle)))
        }
    
    def analyze_pushup(self, landmarks):
        if not landmarks:
            return {"valid": False, "feedback": "No pose detected"}
        shoulder = self.pose_analyzer.get_landmark_coordinates(landmarks, 11)
        elbow = self.pose_analyzer.get_landmark_coordinates(landmarks, 13)
        wrist = self.pose_analyzer.get_landmark_coordinates(landmarks, 15)
        elbow_angle = self.pose_analyzer.calculate_angle(shoulder, elbow, wrist)
        if elbow_angle > 160:
            phase, feedback = "up_position", "Good extension"
        elif elbow_angle < 90:
            phase, feedback = "down_position", "Good depth - now push up!"
        else:
            phase, feedback = "mid_position", "Keep going"
        return {
            "valid": True, "exercise": "pushup", "phase": phase,
            "elbow_angle": elbow_angle, "feedback": feedback,
            "form_score": min(100, max(0, 100 - abs(elbow_angle - 90) / 2))
        }

class SessionAnalyzer:
    @staticmethod
    def detect_repetitions(data, exercise_type):
        """
        Detects repetitions using a state machine (Top -> Bottom -> Top).
        """
        reps = 0
        # State: 'up' (at the top position) or 'down' (at the bottom position)
        current_state = "up" 

        if exercise_type == "squat":
            rep_start_phase = "standing"
            rep_end_phase = "deep_squat"
        elif exercise_type == "pushup":
            rep_start_phase = "up_position"
            rep_end_phase = "down_position"
        else:
            return 0 

        for d in data:
            if not d.get("valid", False): continue
            phase = d.get("phase", "")

            # 1. Transition to the bottom phase 
            if current_state == "up" and phase == rep_end_phase:
                current_state = "down"
                
            # 2. Transition back to the top phase (completes a rep)
            elif current_state == "down" and phase == rep_start_phase:
                reps += 1
                current_state = "up"
                
        return reps 

    @staticmethod
    def get_overall_rating(score):
        if score >= 80: return "EXCELLENT"
        elif score >= 60: return "GOOD"
        elif score >= 40: return "FAIR"
        return "NEEDS IMPROVEMENT"
    
    @staticmethod
    def generate_recommendations(df, exercise_type):
        improvements, strengths = [], []
        avg_form_score = df['form_score'].mean()
        
        if avg_form_score >= 70:
            strengths.append(f"FORM: Excellent form score average ({avg_form_score:.1f})")
        elif avg_form_score < 50:
            improvements.append(f"FORM: Focus on better form (avg: {avg_form_score:.1f})")
            
        if not improvements:
            improvements.append("No specific recommendations — great job!")
            
        return {"improvements": improvements, "strengths": strengths}
    
    @staticmethod
    def generate_text_report(data, duration, exercise_type, calories_burned):
        df = pd.DataFrame([d for d in data if d.get('valid', False)])
        if df.empty: return "No valid data to analyze."
        avg_form_score = df['form_score'].mean()
        reps = SessionAnalyzer.detect_repetitions(data, exercise_type)
        rating = SessionAnalyzer.get_overall_rating(avg_form_score)
        recs = SessionAnalyzer.generate_recommendations(df, exercise_type)
        report = []
        report.append("="*70)
        report.append("EXERCISE SESSION ANALYSIS REPORT")
        report.append("="*70)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Exercise: {exercise_type.upper()}")
        report.append(f"Duration: {duration:.2f} sec")
        report.append(f"Repetitions: {reps}")
        report.append(f"Avg Form Score: {avg_form_score:.2f}")
        report.append(f"Calories Burned: {calories_burned:.2f} kcal")
        report.append(f"Overall Rating: {rating}")
        report.append("")
        report.append("IMPROVEMENTS:")
        for i, imp in enumerate(recs["improvements"], 1):
            report.append(f"{i}. {imp}")
        report.append("")
        report.append("STRENGTHS:")
        for i, s in enumerate(recs["strengths"], 1):
            report.append(f"{i}. {s}")
        report.append("")
        report.append("="*70)
        report.append("Report generated automatically using AI posture analysis")
        return "\n".join(report)

class CalorieEstimator:
    """Estimate calories burned based on METs and user weight."""
    MET_VALUES = {"squat": 5.0, "pushup": 8.0}
    @staticmethod
    def estimate_calories(exercise_type, weight_kg, duration_sec):
        duration_min = duration_sec / 60
        met = CalorieEstimator.MET_VALUES.get(exercise_type, 5.0)
        return met * 3.5 * weight_kg / 200 * duration_min

class LiveVideoProcessor:
    def __init__(self, exercise_type="squat"):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_detector = self.mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

        self.exercise_analyzer = ExerciseFormAnalyzer()
        self.exercise_type = exercise_type
        # session_data is now handled by st.session_state
    
    def extract_landmarks(self, pose_landmarks, shape):
        # We need the UNscaled coordinates (0-1) for angle calculation
        return [[l.x, l.y, l.z] for l in pose_landmarks.landmark]
    
    def process_frame(self, frame):
        # Process frame 
        frame = cv2.flip(frame, 1) # Flip for mirror effect
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb)
        analysis = {"valid": False}
        if results.pose_landmarks:
            # Draw landmarks first
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                             self.mp_pose.POSE_CONNECTIONS)
            
            # Extract landmarks for analysis (scaled 0-1)
            landmarks = self.extract_landmarks(results.pose_landmarks, frame.shape)
            
            # Perform analysis
            analysis = (self.exercise_analyzer.analyze_squat(landmarks)
                        if self.exercise_type == "squat"
                        else self.exercise_analyzer.analyze_pushup(landmarks))
                        
            # Annotate the frame with feedback
            frame = self.annotate_frame(frame, analysis)
        else:
            cv2.putText(frame, "No pose detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
        return frame, analysis
    
    def annotate_frame(self, frame, a):
        color = (0,255,0) if a.get('form_score',0)>=80 else (0,255,255) if a.get('form_score',0)>=60 else (0,0,255)
        cv2.putText(frame, a.get('feedback',''), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Score: {a.get('form_score',0):.1f}", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"{a.get('exercise','')}: {a.get('phase','')}", (30,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return frame

# ---------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------
for key, val in {
    'workout_started': False, 'countdown_active': False, 'session_data': [],
    'start_time': None, 'show_report': False, 'report_text': "",
    'selected_exercise': "squat", 'user_info': {"weight":70.0, "height":170.0, "age":25}
}.items():
    if key not in st.session_state: st.session_state[key] = val

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<h1 class="main-header">💪 AI Gym Posture Trainer</h1>', unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR - USER INPUTS
# ---------------------------------------------------
with st.sidebar:
    st.header("👤 Personal Information")
    weight = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=200.0,
    value=float(st.session_state.user_info["weight"])
)

    height = st.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=220.0,
    value=float(st.session_state.user_info["height"])
)

    age = st.number_input(
    "Age (years)",
    min_value=10,
    max_value=90,
    value=int(st.session_state.user_info["age"])
)

st.session_state.user_info = {"weight": weight, "height": height, "age": age}
st.markdown("---")
st.header("⚙️ Workout Settings")
exercise_type = st.selectbox("Select Exercise", ["squat", "pushup"])
st.session_state.selected_exercise = exercise_type

# ---------------------------------------------------
# SHOW REPORT
# ---------------------------------------------------
if st.session_state.show_report and st.session_state.report_text:
    st.success("✅ Workout Complete! Here's your analysis:")
    st.balloons()
    df = pd.DataFrame([d for d in st.session_state.session_data if d.get('valid', False)])
    if not df.empty and st.session_state.start_time is not None:
        avg_score = df['form_score'].mean()
        duration = time.time() - st.session_state.start_time
        reps = SessionAnalyzer.detect_repetitions(st.session_state.session_data, st.session_state.selected_exercise)
        calories = CalorieEstimator.estimate_calories(st.session_state.selected_exercise, st.session_state.user_info["weight"], duration)
        rating = SessionAnalyzer.get_overall_rating(avg_score)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1: st.metric("Duration", f"{duration:.1f}s")
        with col2: st.metric("Repetitions", reps)
        with col3: st.metric("Frames", len(df))
        with col4: st.metric("Avg Score", f"{avg_score:.1f}")
        with col5: st.metric("Calories", f"{calories:.1f} kcal")
        with col6:
            badge = {"EXCELLENT":"excellent-badge","GOOD":"good-badge"}.get(rating,"needs-improvement-badge")
            st.markdown(f'<span class="{badge}">{rating}</span>', unsafe_allow_html=True)
        st.markdown("---")
    st.download_button("📄 Download Report", st.session_state.report_text,
                        file_name=f"workout_report_{datetime.now():%Y%m%d_%H%M%S}.txt")
    if st.button("🔄 New Workout", use_container_width=True):
        for k in ['session_data','workout_started','countdown_active','start_time','show_report','report_text']:
            st.session_state[k] = [] if k=="session_data" else False if k!="start_time" else None
        st.rerun()

# ---------------------------------------------------
# WORKOUT MODE
# ---------------------------------------------------
else:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if not st.session_state.workout_started and not st.session_state.countdown_active:
            st.info(f"👋 Position yourself for **{st.session_state.selected_exercise.upper()}** and click START when ready!")
            if st.button("🚀 START WORKOUT", use_container_width=True): 
                st.session_state.countdown_active = True
                st.rerun()
        if st.session_state.countdown_active:
            countdown = st.empty()
            for i in range(5,0,-1):
                countdown.markdown(f'<div class="countdown-text">{i}</div>', unsafe_allow_html=True)
                time.sleep(1)
            countdown.empty()
            st.session_state.countdown_active=False
            st.session_state.workout_started=True
            st.session_state.start_time=time.time()
            st.session_state.session_data = [] # Reset data
            st.rerun()
            
    if st.session_state.workout_started:
        st.markdown("### 📹 Live Feed (Camera 0 - Main Analysis)")
        
        # --- Dual Camera Setup ---
        video_col_main, video_col_secondary = st.columns(2)
        with video_col_main:
            video_placeholder_main = st.empty()
        with video_col_secondary:
            st.markdown("### 📹 Live Feed (Camera 1 - Secondary View)")
            video_placeholder_secondary = st.empty()

        stop_btn = st.button("⏹️ STOP WORKOUT")
        processor = LiveVideoProcessor(st.session_state.selected_exercise)
        
        # Initialize both cameras
        cap_main = cv2.VideoCapture(0)  # Primary camera (for analysis)
        cap_secondary = cv2.VideoCapture(1) # Secondary camera (for extra view)
        
        is_main_open = cap_main.isOpened()
        is_secondary_open = cap_secondary.isOpened()
        
        if not is_main_open:
            st.error("❌ Cannot access Camera 0. Analysis cannot proceed.")
            st.session_state.workout_started = False
        elif not is_secondary_open:
            st.warning("⚠️ Camera 1 not found. Showing only the main feed.")
            
        if st.session_state.workout_started:
            # Main video loop
            while st.session_state.workout_started:
                # Read frames from both cameras
                # handling secondary frame 
                ret_main, frame_main = cap_main.read()
                ret_secondary, frame_secondary = cap_secondary.read() if is_secondary_open else (False, None)
                
                if not ret_main: 
                    st.error("❌ Main Camera feed lost.")
                    st.session_state.workout_started = False
                    break
                
                # --- Analysis and Display for Main Camera (Camera 0) ---
                processed_main, analysis = processor.process_frame(frame_main)
                
                # Store valid analysis data
                if analysis.get('valid', False):
                    st.session_state.session_data.append(analysis)
                    
                rgb_main = cv2.cvtColor(processed_main, cv2.COLOR_BGR2RGB)
                video_placeholder_main.image(rgb_main, channels="RGB")
                
                # --- Display for Secondary Camera (Camera 1) ---
                if ret_secondary and frame_secondary is not None:
                    frame_secondary = cv2.flip(frame_secondary, 1) # Flip for consistency
                    rgb_secondary = cv2.cvtColor(frame_secondary, cv2.COLOR_BGR2RGB)
                    video_placeholder_secondary.image(rgb_secondary, channels="RGB")
                elif is_secondary_open and not ret_secondary:
                    st.warning("⚠️ Secondary camera feed lost.")

                if stop_btn:
                    st.session_state.workout_started=False
                    break
                
                time.sleep(0.01) # Small sleep to manage CPU usage
                
            # Cleanup after loop ends
            cap_main.release()
            if is_secondary_open:
                cap_secondary.release()
            
        if not st.session_state.workout_started:
            # Generate final report and trigger report view
            duration = time.time() - st.session_state.start_time
            calories = CalorieEstimator.estimate_calories(st.session_state.selected_exercise,
                                                          st.session_state.user_info["weight"], duration)
            report = SessionAnalyzer.generate_text_report(st.session_state.session_data, duration,
                                                          st.session_state.selected_exercise, calories)
            st.session_state.report_text = report
            st.session_state.show_report = True
            st.rerun()

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:#888;'>Made with ❤️ using Streamlit, MediaPipe & OpenCV</p>",
             unsafe_allow_html=True)