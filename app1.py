import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from datetime import datetime
from typing import Dict, Any, Tuple, List
import io

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

class SessionAnalyzer:
    """Comprehensive session analysis and report generation"""
    
    @staticmethod
    def detect_repetitions(session_data: List[Dict], exercise_type: str) -> int:
        """Detect number of repetitions based on phase transitions"""
        if not session_data:
            return 0
        
        reps = 0
        prev_phase = None
        
        for data in session_data:
            if not data.get('valid', False):
                continue
            
            phase = data.get('phase', '')
            
            if exercise_type == 'squat':
                # Count transition from deep_squat back to standing
                if prev_phase == 'deep_squat' and phase == 'standing':
                    reps += 1
            elif exercise_type == 'pushup':
                # Count transition from down_position back to up_position
                if prev_phase == 'down_position' and phase == 'up_position':
                    reps += 1
            
            prev_phase = phase
        
        return max(reps, 1)  # At least 1 rep if data exists
    
    @staticmethod
    def calculate_rom_consistency(angles: List[float]) -> float:
        """Calculate range of motion consistency"""
        if len(angles) < 2:
            return 0
        return (np.std(angles) / np.mean(angles)) * 100 if np.mean(angles) > 0 else 0
    
    @staticmethod
    def get_overall_rating(avg_form_score: float) -> str:
        """Determine overall rating"""
        if avg_form_score >= 80:
            return "EXCELLENT"
        elif avg_form_score >= 60:
            return "GOOD"
        elif avg_form_score >= 40:
            return "FAIR"
        else:
            return "NEEDS IMPROVEMENT"
    
    @staticmethod
    def generate_recommendations(df: pd.DataFrame, exercise_type: str) -> Dict[str, List[str]]:
        """Generate personalized recommendations"""
        improvements = []
        strengths = []
        
        avg_form_score = df['form_score'].mean()
        angle_col = 'knee_angle' if exercise_type == 'squat' else 'elbow_angle'
        
        if angle_col in df.columns:
            angles = df[angle_col].values
            angle_std = np.std(angles)
            angle_mean = np.mean(angles)
            
            # Depth analysis
            if exercise_type == 'squat':
                deep_squats = len(df[df['phase'] == 'deep_squat'])
                total_valid = len(df[df['valid'] == True])
                
                if deep_squats / total_valid > 0.3:
                    strengths.append(f"DEPTH: Excellent depth consistency - {deep_squats} frames below 90°")
                elif deep_squats / total_valid < 0.1:
                    improvements.append("DEPTH: Try to achieve deeper squats (below 90° knee angle)")
                
                if angle_mean > 130:
                    improvements.append("DEPTH: Average squat depth is shallow - aim for deeper squats")
                    
            elif exercise_type == 'pushup':
                down_pushups = len(df[df['phase'] == 'down_position'])
                total_valid = len(df[df['valid'] == True])
                
                if down_pushups / total_valid > 0.2:
                    strengths.append(f"DEPTH: Good push-up depth achieved in {down_pushups} frames")
                else:
                    improvements.append("DEPTH: Try to lower chest closer to ground")
            
            # Consistency analysis
            rom_consistency = (angle_std / angle_mean) * 100 if angle_mean > 0 else 0
            if rom_consistency < 15:
                strengths.append(f"CONSISTENCY: Excellent ROM consistency (variation: {rom_consistency:.1f}%)")
            elif rom_consistency > 30:
                improvements.append(f"CONSISTENCY: Work on consistent range of motion (variation: {rom_consistency:.1f}%)")
        
        # Form score analysis
        if avg_form_score >= 70:
            strengths.append(f"FORM: Excellent form score average ({avg_form_score:.1f})")
        elif avg_form_score < 50:
            improvements.append(f"FORM: Focus on proper form technique (current avg: {avg_form_score:.1f})")
        
        # Tempo analysis
        phase_changes = df['phase'].ne(df['phase'].shift()).sum()
        if phase_changes > 6:
            strengths.append("TEMPO: Good repetition tempo control")
        elif phase_changes < 3:
            improvements.append("TEMPO: Try to maintain steady tempo throughout exercise")
        
        # Smoothness
        if angle_col in df.columns:
            angle_diff = np.diff(angles)
            if np.mean(np.abs(angle_diff)) < 5:
                strengths.append("SMOOTHNESS: Movement appears smooth and controlled")
            else:
                improvements.append("SMOOTHNESS: Work on smoother transitions between positions")
        
        if not improvements:
            improvements.append("No specific recommendations - great job!")
        
        return {"improvements": improvements, "strengths": strengths}
    
    @staticmethod
    def generate_text_report(session_data: List[Dict], duration: float, exercise_type: str) -> str:
        """Generate comprehensive text report"""
        df = pd.DataFrame([d for d in session_data if d.get('valid', False)])
        
        if df.empty:
            return "No valid data to analyze."
        
        # Basic statistics
        total_frames = len(df)
        avg_form_score = df['form_score'].mean()
        reps = SessionAnalyzer.detect_repetitions(session_data, exercise_type)
        rating = SessionAnalyzer.get_overall_rating(avg_form_score)
        
        # Phase distribution
        phase_counts = df['phase'].value_counts()
        
        # Angle statistics
        angle_col = 'knee_angle' if exercise_type == 'squat' else 'elbow_angle'
        angle_stats = df[angle_col].describe() if angle_col in df.columns else None
        
        # Recommendations
        recommendations = SessionAnalyzer.generate_recommendations(df, exercise_type)
        
        # Build report
        report = []
        report.append("=" * 70)
        report.append("EXERCISE SESSION ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Exercise Type: {exercise_type.upper()}")
        report.append("")
        
        report.append("SESSION OVERVIEW:")
        report.append("-" * 70)
        report.append(f"Total Frames: {total_frames}")
        report.append(f"Valid Frames: {total_frames} (100.0%)")
        report.append(f"Session Duration: {duration:.2f} seconds")
        report.append(f"Detected Repetitions: {reps}")
        report.append(f"Average Form Score: {avg_form_score:.2f}")
        report.append("")
        
        report.append("EXERCISE TYPES:")
        report.append("-" * 70)
        report.append(f"{exercise_type}: {total_frames} frames")
        report.append("")
        
        report.append("PHASES:")
        report.append("-" * 70)
        for phase, count in phase_counts.items():
            report.append(f"{phase}: {count} frames")
        report.append("")
        
        report.append("OVERALL RATING:")
        report.append("-" * 70)
        report.append(f"{rating}")
        report.append("")
        report.append("")
        
        report.append("=" * 70)
        report.append("DETAILED ANALYSIS & RECOMMENDATIONS")
        report.append("=" * 70)
        report.append("")
        
        report.append("AREAS FOR IMPROVEMENT:")
        report.append("-" * 70)
        for i, improvement in enumerate(recommendations['improvements'], 1):
            report.append(f"{i}. {improvement}")
        report.append("")
        
        report.append("STRENGTHS IDENTIFIED:")
        report.append("-" * 70)
        for i, strength in enumerate(recommendations['strengths'], 1):
            report.append(f"{i}. {strength}")
        report.append("")
        report.append("")
        
        if angle_stats is not None:
            report.append("STATISTICAL SUMMARY:")
            report.append("-" * 70)
            angle_name = "Knee Angle" if exercise_type == 'squat' else "Elbow Angle"
            report.append(f"{angle_name} Statistics:")
            report.append(f"  Mean: {angle_stats['mean']:.2f}°")
            report.append(f"  Std Dev: {angle_stats['std']:.2f}°")
            report.append(f"  Min: {angle_stats['min']:.2f}°")
            report.append(f"  Max: {angle_stats['max']:.2f}°")
            report.append("")
            
            report.append(f"Form Score Statistics:")
            form_stats = df['form_score'].describe()
            report.append(f"  Mean: {form_stats['mean']:.2f}")
            report.append(f"  Std Dev: {form_stats['std']:.2f}")
            report.append(f"  Min: {form_stats['min']:.2f}")
            report.append(f"  Max: {form_stats['max']:.2f}")
            report.append("")
        
        report.append("NEXT STEPS:")
        report.append("-" * 70)
        report.append("1. Focus on the highlighted improvement areas")
        report.append("2. Maintain consistency in good aspects")
        report.append("3. Track progress over multiple sessions")
        report.append("4. Consider video review for technique refinement")
        report.append("")
        report.append("=" * 70)
        report.append("Report generated automatically using Python analytics")
        report.append("=" * 70)
        
        return "\n".join(report)

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
if 'show_report' not in st.session_state:
    st.session_state.show_report = False
if 'report_text' not in st.session_state:
    st.session_state.report_text = ""
if 'selected_exercise' not in st.session_state:
    st.session_state.selected_exercise = "squat"

# Header
st.markdown('<h1 class="main-header">💪 AI Gym Posture Trainer</h1>', unsafe_allow_html=True)

# Show report if available
if st.session_state.show_report and st.session_state.report_text:
    st.success("✅ Workout Complete! Here's your comprehensive analysis:")
    st.balloons()
    
    # Parse report for display
    df = pd.DataFrame([d for d in st.session_state.session_data if d.get('valid', False)])
    
    if not df.empty:
        avg_form_score = df['form_score'].mean()
        rating = SessionAnalyzer.get_overall_rating(avg_form_score)
        duration = time.time() - st.session_state.start_time if st.session_state.start_time else 0
        reps = SessionAnalyzer.detect_repetitions(st.session_state.session_data, st.session_state.selected_exercise)
        
        # Overview metrics
        st.markdown("### 📊 Session Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Duration", f"{duration:.1f}s")
        with col2:
            st.metric("Repetitions", reps)
        with col3:
            st.metric("Total Frames", len(df))
        with col4:
            st.metric("Avg Form Score", f"{avg_form_score:.1f}")
        with col5:
            if rating == "EXCELLENT":
                st.markdown('<span class="excellent-badge">EXCELLENT</span>', unsafe_allow_html=True)
            elif rating == "GOOD":
                st.markdown('<span class="good-badge">GOOD</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="needs-improvement-badge">NEEDS WORK</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed analysis
        recommendations = SessionAnalyzer.generate_recommendations(df, st.session_state.selected_exercise)
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### ✅ Strengths Identified")
            st.markdown('<div class="report-section">', unsafe_allow_html=True)
            for strength in recommendations['strengths']:
                st.markdown(f"- {strength}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_right:
            st.markdown("### 🎯 Areas for Improvement")
            st.markdown('<div class="report-section">', unsafe_allow_html=True)
            for improvement in recommendations['improvements']:
                st.markdown(f"- {improvement}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Statistical summary
        st.markdown("### 📈 Statistical Summary")
        angle_col = 'knee_angle' if st.session_state.selected_exercise == 'squat' else 'elbow_angle'
        
        if angle_col in df.columns:
            stat_col1, stat_col2 = st.columns(2)
            
            with stat_col1:
                st.markdown("#### Joint Angle Statistics")
                angle_stats = df[angle_col].describe()
                angle_name = "Knee Angle" if st.session_state.selected_exercise == 'squat' else "Elbow Angle"
                st.write(f"**{angle_name}:**")
                st.write(f"- Mean: {angle_stats['mean']:.2f}°")
                st.write(f"- Std Dev: {angle_stats['std']:.2f}°")
                st.write(f"- Range: {angle_stats['min']:.2f}° - {angle_stats['max']:.2f}°")
            
            with stat_col2:
                st.markdown("#### Form Score Statistics")
                form_stats = df['form_score'].describe()
                st.write("**Form Score:**")
                st.write(f"- Mean: {form_stats['mean']:.2f}")
                st.write(f"- Std Dev: {form_stats['std']:.2f}")
                st.write(f"- Range: {form_stats['min']:.2f} - {form_stats['max']:.2f}")
        
        st.markdown("---")
        
        # Phase distribution
        st.markdown("### 📋 Phase Distribution")
        phase_counts = df['phase'].value_counts()
        phase_df = pd.DataFrame({
            'Phase': phase_counts.index,
            'Frames': phase_counts.values,
            'Percentage': (phase_counts.values / len(df) * 100).round(1)
        })
        st.dataframe(phase_df, use_container_width=True)
        
        st.markdown("---")
        
        # Download buttons
        st.markdown("### 📥 Download Reports")
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            # Download text report
            st.download_button(
                label="📄 Download Text Report",
                data=st.session_state.report_text,
                file_name=f"workout_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_dl2:
            # Download CSV data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Data",
                data=csv,
                file_name=f"workout_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Full text report in expander
        with st.expander("📋 View Full Text Report"):
            st.code(st.session_state.report_text, language=None)
    
    # New workout button
    if st.button("🔄 Start New Workout", use_container_width=True):
        st.session_state.session_data = []
        st.session_state.workout_started = False
        st.session_state.countdown_active = False
        st.session_state.start_time = None
        st.session_state.show_report = False
        st.session_state.report_text = ""
        st.rerun()
    
else:
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        exercise_type = st.selectbox(
            "Select Exercise",
            ["squat", "pushup"],
            help="Choose the exercise you want to perform"
        )
        st.session_state.selected_exercise = exercise_type
        
        st.markdown("---")
        st.subheader("📊 Live Session Stats")
        
        if st.session_state.session_data and st.session_state.workout_started:
            valid_data = [d for d in st.session_state.session_data if d.get('valid', False)]
            if valid_data:
                avg_score = np.mean([d.get('form_score', 0) for d in valid_data])
                st.metric("Average Form Score", f"{avg_score:.1f}/100")
                st.metric("Total Frames Analyzed", len(st.session_state.session_data))
                
                if st.session_state.start_time:
                    duration = time.time() - st.session_state.start_time
                    st.metric("Session Duration", f"{int(duration)}s")
                
                # Live rep counter
                reps = SessionAnalyzer.detect_repetitions(st.session_state.session_data, exercise_type)
                st.metric("Repetitions", reps)
        
        st.markdown("---")
        
        if st.button("🔄 Reset Session"):
            st.session_state.session_data = []
            st.session_state.workout_started = False
            st.session_state.countdown_active = False
            st.session_state.start_time = None
            st.session_state.show_report = False
            st.session_state.report_text = ""
            st.rerun()
    
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
            
            # Generate comprehensive report
            duration = time.time() - st.session_state.start_time if st.session_state.start_time else 0
            report_text = SessionAnalyzer.generate_text_report(
                st.session_state.session_data, 
                duration, 
                st.session_state.selected_exercise
            )
            st.session_state.report_text = report_text
            st.session_state.show_report = True
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