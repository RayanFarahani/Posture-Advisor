import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from collections import deque
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class PostureMetrics:
    """Advanced posture metrics calculator"""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def get_comprehensive_metrics(self, landmarks, width, height):
        """Calculate all posture metrics"""
        try:
            metrics = {}
            
            # Key landmarks
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            
            # 1. Spine alignment (torso angle from vertical)
            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2 * width
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2 * height
            hip_mid_x = (left_hip.x + right_hip.x) / 2 * width
            hip_mid_y = (left_hip.y + right_hip.y) / 2 * height
            
            spine_angle = np.arctan2(shoulder_mid_x - hip_mid_x, hip_mid_y - shoulder_mid_y) * 180 / np.pi
            metrics['spine_angle'] = spine_angle
            metrics['spine_points'] = [(int(shoulder_mid_x), int(shoulder_mid_y)), 
                                      (int(hip_mid_x), int(hip_mid_y))]
            
            # 2. Forward head posture (craniovertebral angle)
            ear_mid_x = (left_ear.x + right_ear.x) / 2 * width
            ear_mid_y = (left_ear.y + right_ear.y) / 2 * height
            
            # Calculate angle: ear -> shoulder -> vertical
            cv_angle = self.calculate_angle(
                [ear_mid_x, ear_mid_y],
                [shoulder_mid_x, shoulder_mid_y],
                [shoulder_mid_x, shoulder_mid_y + 100]  # vertical reference
            )
            metrics['cv_angle'] = cv_angle
            metrics['head_forward_distance'] = abs(ear_mid_x - shoulder_mid_x)
            metrics['head_points'] = [(int(ear_mid_x), int(ear_mid_y)), 
                                     (int(shoulder_mid_x), int(shoulder_mid_y))]
            
            # 3. Shoulder alignment
            shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y) * height
            shoulder_angle = np.arctan2(right_shoulder.y - left_shoulder.y, 
                                       right_shoulder.x - left_shoulder.x) * 180 / np.pi
            metrics['shoulder_height_diff'] = shoulder_height_diff
            metrics['shoulder_angle'] = abs(shoulder_angle)
            
            # 4. Neck angle (head tilt)
            nose_y = nose.y * height
            ear_mid_y_norm = ear_mid_y
            neck_angle = abs(nose_y - ear_mid_y_norm)
            metrics['neck_angle'] = neck_angle
            
            # 5. Upper body symmetry
            left_shoulder_pos = np.array([left_shoulder.x * width, left_shoulder.y * height])
            right_shoulder_pos = np.array([right_shoulder.x * width, right_shoulder.y * height])
            left_hip_pos = np.array([left_hip.x * width, left_hip.y * height])
            right_hip_pos = np.array([right_hip.x * width, right_hip.y * height])
            
            left_torso_length = np.linalg.norm(left_shoulder_pos - left_hip_pos)
            right_torso_length = np.linalg.norm(right_shoulder_pos - right_hip_pos)
            symmetry_diff = abs(left_torso_length - right_torso_length)
            metrics['symmetry_diff'] = symmetry_diff
            
            # 6. Sitting height consistency
            metrics['shoulder_height'] = shoulder_mid_y
            metrics['hip_height'] = hip_mid_y
            
            # 7. Confidence score (visibility of key landmarks)
            visibility_scores = [
                nose.visibility, left_ear.visibility, right_ear.visibility,
                left_shoulder.visibility, right_shoulder.visibility,
                left_hip.visibility, right_hip.visibility
            ]
            metrics['confidence'] = np.mean(visibility_scores)
            
            return metrics
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None

class PostureAnalyzer:
    """Intelligent posture analysis with ML-like behavior"""
    def __init__(self):
        self.baseline = None
        self.history = {
            'spine_angle': deque(maxlen=300),  # 10 seconds at 30fps
            'cv_angle': deque(maxlen=300),
            'shoulder_diff': deque(maxlen=300),
            'timestamps': deque(maxlen=300)
        }
        
        # Adaptive thresholds
        self.thresholds = {
            'spine_deviation': 12,
            'cv_angle_deviation': 12,
            'head_forward': 50,
            'shoulder_tilt': 25,
            'neck_angle_deviation': 25
        }
        
        # Session statistics
        self.session_start = time.time()
        self.posture_scores = []
        self.poor_posture_duration = 0
        self.good_posture_duration = 0
        self.last_score_time = time.time()
        
        # Issue tracking for smarter feedback
        self.issue_persistence = {
            'spine_forward': deque(maxlen=60),
            'spine_back': deque(maxlen=60),
            'head_forward': deque(maxlen=60),
            'shoulder_uneven': deque(maxlen=60),
            'neck_strain': deque(maxlen=60)
        }
        
    def calibrate(self, metrics_list):
        """Calibrate baseline from multiple frames"""
        # Use median to reduce outlier effect
        self.baseline = {
            'spine_angle': np.median([m['spine_angle'] for m in metrics_list]),
            'cv_angle': np.median([m['cv_angle'] for m in metrics_list]),
            'head_forward_distance': np.median([m['head_forward_distance'] for m in metrics_list]),
            'shoulder_height_diff': np.median([m['shoulder_height_diff'] for m in metrics_list]),
            'shoulder_angle': np.median([m['shoulder_angle'] for m in metrics_list]),
            'neck_angle': np.median([m['neck_angle'] for m in metrics_list]),
            'shoulder_height': np.median([m['shoulder_height'] for m in metrics_list]),
            'symmetry_diff': np.median([m['symmetry_diff'] for m in metrics_list])
        }
        
        # Calculate standard deviations for adaptive thresholds
        spine_std = np.std([m['spine_angle'] for m in metrics_list])
        cv_std = np.std([m['cv_angle'] for m in metrics_list])
        
        # Set thresholds as multiples of baseline variations
        # These represent "significant" deviations from the user's calibrated position
        self.thresholds['spine_deviation'] = max(8, spine_std * 3)
        self.thresholds['cv_angle_deviation'] = max(10, cv_std * 3)
        self.thresholds['head_forward'] = 35  # Pixels
        self.thresholds['shoulder_tilt'] = self.baseline['shoulder_angle'] + 15
        self.thresholds['neck_angle_deviation'] = 25
        
    def analyze(self, metrics):
        """Comprehensive posture analysis"""
        if not metrics or not self.baseline:
            return None
        
        # Update history
        current_time = time.time()
        self.history['spine_angle'].append(metrics['spine_angle'])
        self.history['cv_angle'].append(metrics['cv_angle'])
        self.history['shoulder_diff'].append(metrics['shoulder_height_diff'])
        self.history['timestamps'].append(current_time)
        
        # Calculate ABSOLUTE deviations from user's calibrated baseline
        spine_dev = abs(metrics['spine_angle'] - self.baseline['spine_angle'])
        cv_dev = abs(metrics['cv_angle'] - self.baseline['cv_angle'])
        head_forward = abs(metrics['head_forward_distance'] - self.baseline['head_forward_distance'])
        shoulder_tilt = abs(metrics['shoulder_angle'] - self.baseline['shoulder_angle'])
        neck_dev = abs(metrics['neck_angle'] - self.baseline['neck_angle'])
        shoulder_height_dev = abs(metrics['shoulder_height_diff'] - self.baseline['shoulder_height_diff'])
        
        # Detect issues with persistence tracking
        issues = []
        severity_scores = {}
        
        # Spine deviation (forward OR backward lean from baseline)
        if spine_dev > self.thresholds['spine_deviation']:
            # Determine direction
            if metrics['spine_angle'] > self.baseline['spine_angle']:
                issue_name = 'spine_forward'
            else:
                issue_name = 'spine_back'
            
            issues.append(issue_name)
            # Score based on how far from baseline (0-35 points penalty)
            severity_scores[issue_name] = min(35, (spine_dev / self.thresholds['spine_deviation']) * 35)
            self.issue_persistence[issue_name].append(1)
            
            # Clear the opposite direction
            opposite = 'spine_back' if issue_name == 'spine_forward' else 'spine_forward'
            self.issue_persistence[opposite].append(0)
        else:
            self.issue_persistence['spine_forward'].append(0)
            self.issue_persistence['spine_back'].append(0)
        
        # CV angle deviation (head position changed from baseline)
        if cv_dev > self.thresholds['cv_angle_deviation']:
            issues.append('head_forward')
            severity_scores['head_forward'] = min(30, (cv_dev / self.thresholds['cv_angle_deviation']) * 30)
            self.issue_persistence['head_forward'].append(1)
        else:
            self.issue_persistence['head_forward'].append(0)
        
        # Shoulder tilt deviation from baseline
        if shoulder_tilt > self.thresholds['shoulder_tilt']:
            issues.append('shoulder_uneven')
            severity_scores['shoulder_uneven'] = min(20, (shoulder_tilt / self.thresholds['shoulder_tilt']) * 20)
            self.issue_persistence['shoulder_uneven'].append(1)
        else:
            self.issue_persistence['shoulder_uneven'].append(0)
        
        # Neck angle deviation
        if neck_dev > self.thresholds['neck_angle_deviation']:
            issues.append('neck_strain')
            severity_scores['neck_strain'] = min(15, (neck_dev / self.thresholds['neck_angle_deviation']) * 15)
            self.issue_persistence['neck_strain'].append(1)
        else:
            self.issue_persistence['neck_strain'].append(0)
        
        # Calculate overall posture score (0-100)
        # Start at 100, subtract severity penalties
        posture_score = 100
        for issue, score in severity_scores.items():
            posture_score -= score
        posture_score = max(0, posture_score)
        
        # Track session statistics
        time_delta = current_time - self.last_score_time
        if posture_score >= 70:
            self.good_posture_duration += time_delta
        else:
            self.poor_posture_duration += time_delta
        self.last_score_time = current_time
        self.posture_scores.append(posture_score)
        
        # Prioritize issues by persistence (must be present >50% of recent frames)
        persistent_issues = []
        for issue in issues:
            persistence = sum(self.issue_persistence[issue]) / len(self.issue_persistence[issue])
            if persistence > 0.5:
                persistent_issues.append((issue, persistence, severity_scores.get(issue, 0)))
        
        # Sort by severity
        persistent_issues.sort(key=lambda x: x[2], reverse=True)
        
        return {
            'score': posture_score,
            'issues': [i[0] for i in persistent_issues],
            'issue_details': persistent_issues,
            'deviations': {
                'spine': spine_dev,
                'cv_angle': cv_dev,
                'head_forward': head_forward,
                'shoulder_tilt': shoulder_tilt,
                'neck': neck_dev
            },
            'raw_values': {
                'spine_angle': metrics['spine_angle'],
                'cv_angle': metrics['cv_angle'],
                'baseline_spine': self.baseline['spine_angle'],
                'baseline_cv': self.baseline['cv_angle']
            },
            'metrics': metrics
        }
    
    def get_session_stats(self):
        """Get comprehensive session statistics"""
        session_duration = time.time() - self.session_start
        avg_score = np.mean(self.posture_scores) if self.posture_scores else 0
        
        return {
            'duration': session_duration,
            'average_score': avg_score,
            'good_posture_time': self.good_posture_duration,
            'poor_posture_time': self.poor_posture_duration,
            'good_posture_percent': (self.good_posture_duration / session_duration * 100) if session_duration > 0 else 0
        }

class FeedbackSystem:
    """Intelligent feedback with personalization"""
    def __init__(self):
        self.last_feedback_time = 0
        self.feedback_cooldown = 20  # Increased for less interruption
        self.feedback_history = []
        self.user_responsiveness = 1.0  # Track how quickly user corrects
        
        self.messages = {
            'spine_forward': [
                "💡 Gentle reminder: Sit back in your chair",
                "🪑 Try pressing your back against the chair",
                "⬆️ Sit upright - you're leaning forward a bit",
            ],
            'spine_back': [
                "💡 You're leaning back - try sitting upright",
                "🎯 Engage your core and sit forward slightly",
            ],
            'head_forward': [
                "👤 Pull your head back over your shoulders",
                "💡 Imagine a string pulling your head upward",
                "📱 Are you leaning toward the screen? Sit back!",
            ],
            'shoulder_uneven': [
                "⚖️ Level your shoulders - check your posture",
                "💡 Are you leaning on one side? Center yourself",
                "🔄 Adjust your position - shoulders look uneven",
            ],
            'neck_strain': [
                "😌 Relax your neck - you might be straining",
                "💡 Lower your shoulders and lengthen your neck",
            ]
        }
        
        self.encouragements = [
            "✨ Great posture! Keep it up!",
            "🌟 Excellent - you're sitting beautifully!",
            "💪 Perfect posture right now!",
            "👍 That's the way! Stay comfortable!",
        ]
    
    def get_feedback(self, analysis):
        """Generate intelligent, context-aware feedback"""
        current_time = time.time()
        
        # Encouragement for good posture
        if analysis['score'] >= 85 and (current_time - self.last_feedback_time) > 60:
            feedback = np.random.choice(self.encouragements)
            self.last_feedback_time = current_time
            return feedback, 'positive'
        
        # Issue-based feedback with cooldown
        if analysis['issues'] and (current_time - self.last_feedback_time) > self.feedback_cooldown:
            # Get most severe issue
            primary_issue = analysis['issues'][0]
            
            # Vary messages
            message = np.random.choice(self.messages[primary_issue])
            
            # Add severity context
            severity = analysis['issue_details'][0][2] if analysis['issue_details'] else 0
            if severity > 70:
                message = "⚠️ " + message + " (significant deviation)"
            
            self.last_feedback_time = current_time
            self.feedback_history.append((current_time, primary_issue, analysis['score']))
            
            return message, 'corrective'
        
        return None, None

class VisualizationEngine:
    """Advanced visualization and UI"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.graph_data = {
            'scores': deque(maxlen=300),
            'timestamps': deque(maxlen=300)
        }
        
    def draw_metrics_panel(self, frame, analysis, stats):
        """Draw comprehensive metrics panel"""
        panel_width = 400
        panel_height = 250
        panel_x = self.width - panel_width - 10
        panel_y = 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (100, 100, 100), 2)
        
        y_offset = panel_y + 30
        
        # Title
        cv2.putText(frame, "POSTURE METRICS", (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 35
        
        # Posture Score with color coding
        score = analysis['score']
        if score >= 80:
            score_color = (0, 255, 0)
            status = "EXCELLENT"
        elif score >= 60:
            score_color = (0, 255, 255)
            status = "GOOD"
        elif score >= 40:
            score_color = (0, 165, 255)
            status = "FAIR"
        else:
            score_color = (0, 0, 255)
            status = "POOR"
        
        cv2.putText(frame, f"Score: {score:.1f}/100 - {status}", 
                   (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)
        y_offset += 30
        
        # Progress bar for score
        bar_width = panel_width - 40
        bar_height = 20
        bar_x = panel_x + 20
        
        cv2.rectangle(frame, (bar_x, y_offset), 
                     (bar_x + bar_width, y_offset + bar_height), 
                     (60, 60, 60), -1)
        filled_width = int((score / 100) * bar_width)
        cv2.rectangle(frame, (bar_x, y_offset), 
                     (bar_x + filled_width, y_offset + bar_height), 
                     score_color, -1)
        y_offset += 35
        
        # Deviations
        devs = analysis['deviations']
        cv2.putText(frame, f"Spine Dev: {devs['spine']:.1f}deg", 
                   (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 22
        
        cv2.putText(frame, f"Head Dev: {devs['cv_angle']:.1f}deg", 
                   (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 22
        
        cv2.putText(frame, f"Shoulder Dev: {devs['shoulder_tilt']:.1f}deg", 
                   (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 22
        
        cv2.putText(frame, f"Neck Dev: {devs['neck']:.1f}deg", 
                   (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 30
        
        # Session stats
        cv2.line(frame, (panel_x + 10, y_offset), 
                (panel_x + panel_width - 10, y_offset), (100, 100, 100), 1)
        y_offset += 20
        
        duration_min = int(stats['duration'] / 60)
        cv2.putText(frame, f"Session: {duration_min}min", 
                   (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_offset += 22
        
        cv2.putText(frame, f"Good Posture: {stats['good_posture_percent']:.0f}%", 
                   (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    def draw_skeleton_overlay(self, frame, metrics):
        """Draw enhanced skeleton with posture indicators"""
        if not metrics:
            return
        
        # Spine line with color coding
        if 'spine_points' in metrics:
            shoulder, hip = metrics['spine_points']
            spine_color = (0, 255, 0)  # Will be updated based on angle
            cv2.line(frame, shoulder, hip, spine_color, 4)
            cv2.circle(frame, shoulder, 8, (255, 0, 255), -1)
            cv2.circle(frame, hip, 8, (255, 0, 255), -1)
        
        # Head forward indicator
        if 'head_points' in metrics:
            ear, shoulder = metrics['head_points']
            cv2.line(frame, ear, shoulder, (0, 165, 255), 3)
            cv2.circle(frame, ear, 6, (0, 255, 255), -1)
    
    def draw_feedback_banner(self, frame, feedback, feedback_type):
        """Draw attractive feedback banner"""
        if not feedback:
            return
        
        banner_height = 100
        banner_y = (self.height - banner_height) // 2
        
        # Background with gradient effect
        overlay = frame.copy()
        color = (50, 150, 50) if feedback_type == 'positive' else (50, 100, 150)
        cv2.rectangle(overlay, (0, banner_y), 
                     (self.width, banner_y + banner_height), 
                     color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        border_color = (100, 255, 100) if feedback_type == 'positive' else (100, 200, 255)
        cv2.rectangle(frame, (0, banner_y), 
                     (self.width, banner_y + banner_height), 
                     border_color, 3)
        
        # Text
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = banner_y + (banner_height + text_size[1]) // 2
        
        # Shadow effect
        cv2.putText(frame, feedback, (text_x + 2, text_y + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, feedback, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    def update_graph_data(self, score, timestamp):
        """Update real-time graph data"""
        self.graph_data['scores'].append(score)
        self.graph_data['timestamps'].append(timestamp)

# Main application
class AdvancedPostureAdvisor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.metrics_calculator = PostureMetrics()
        self.analyzer = PostureAnalyzer()
        self.feedback_system = FeedbackSystem()
        
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to access camera")
        
        self.height, self.width, _ = frame.shape
        self.visualizer = VisualizationEngine(self.width, self.height)
        
        self.calibrated = False
        self.calibration_frames = []
        self.current_feedback = None
        self.feedback_type = None
        self.feedback_start_time = 0
        self.feedback_duration = 6
        
        self.paused = False
        
    def calibrate(self):
        """Perform calibration"""
        if len(self.calibration_frames) >= 60:
            self.analyzer.calibrate(self.calibration_frames)
            self.calibrated = True
            print("\n" + "="*60)
            print("✓ CALIBRATION COMPLETE!")
            print("="*60)
            print(f"Baseline spine angle: {self.analyzer.baseline['spine_angle']:.1f}°")
            print(f"Baseline CV angle: {self.analyzer.baseline['cv_angle']:.1f}°")
            print("\n📊 Monitoring started... Maintain good posture!\n")
            return True
        return False
    
    def run(self):
        """Main application loop"""
        print("╔" + "="*60 + "╗")
        print("║" + " "*15 + "ADVANCED POSTURE ADVISOR" + " "*22 + "║")
        print("╚" + "="*60 + "╝")
        print("\n🎯 Professional Ergonomic Monitoring System")
        print("\n📋 How It Works:")
        print("  • YOU choose your comfortable sitting position")
        print("  • System learns YOUR baseline (not generic standards)")
        print("  • Alerts you when you deviate from YOUR position")
        print("  • Score starts at 100, decreases with deviations")
        print("\n📋 Setup Instructions:")
        print("  1. Sit in YOUR most comfortable, balanced position")
        print("  2. Face camera directly, ensure full upper body visible")
        print("  3. Hold steady and press SPACE to calibrate (2 seconds)")
        print("  4. System monitors deviations from YOUR baseline")
        print("\n⌨️  Controls:")
        print("  SPACE  - Calibrate/Recalibrate baseline posture")
        print("  P      - Pause/Resume monitoring")
        print("  S      - Save session statistics")
        print("  R      - Reset session")
        print("  Q      - Quit")
        print("\n⏳ Position yourself and press SPACE when ready...\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if not self.paused:
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # Draw pose landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
                    
                    # Calculate metrics
                    metrics = self.metrics_calculator.get_comprehensive_metrics(
                        results.pose_landmarks.landmark, self.width, self.height
                    )
                    
                    if metrics and metrics['confidence'] > 0.5:
                        if not self.calibrated:
                            # Calibration mode
                            self.calibration_frames.append(metrics)
                            if len(self.calibration_frames) > 90:
                                self.calibration_frames.pop(0)
                            
                            progress = len(self.calibration_frames) / 60 * 100
                            cv2.putText(frame, f"Calibration: {progress:.0f}% - Hold steady!", 
                                       (10, self.height - 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        else:
                            # Analysis mode
                            analysis = self.analyzer.analyze(metrics)
                            
                            if analysis:
                                # Update visualization
                                self.visualizer.draw_skeleton_overlay(frame, metrics)
                                stats = self.analyzer.get_session_stats()
                                self.visualizer.draw_metrics_panel(frame, analysis, stats)
                                self.visualizer.update_graph_data(analysis['score'], time.time())
                                
                                # Get feedback
                                feedback, ftype = self.feedback_system.get_feedback(analysis)
                                if feedback:
                                    self.current_feedback = feedback
                                    self.feedback_type = ftype
                                    self.feedback_start_time = time.time()
                                
                                # Display feedback
                                if self.current_feedback and \
                                   (time.time() - self.feedback_start_time) < self.feedback_duration:
                                    self.visualizer.draw_feedback_banner(
                                        frame, self.current_feedback, self.feedback_type
                                    )
                else:
                    if not self.calibrated:
                        cv2.putText(frame, "⚠️ Position yourself in camera view", 
                                   (10, self.height - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # Paused overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, "⏸ PAUSED - Press P to resume", 
                           (self.width // 2 - 200, self.height // 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            cv2.imshow('Advanced Posture Advisor', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not self.calibrated:
                    if self.calibrate():
                        pass
                else:
                    # Recalibrate
                    self.calibration_frames = []
                    self.calibrated = False
                    print("\n🔄 Recalibrating... Hold your best posture")
            elif key == ord('p'):
                self.paused = not self.paused
                status = "paused" if self.paused else "resumed"
                print(f"\n⏸️  Monitoring {status}")
            elif key == ord('s'):
                self.save_session_stats()
            elif key == ord('r'):
                self.reset_session()
        
        self.cleanup()
    
    def save_session_stats(self):
        """Save session statistics to file"""
        if not self.calibrated:
            print("⚠️  Cannot save - not calibrated yet")
            return
        
        stats = self.analyzer.get_session_stats()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'timestamp': timestamp,
            'session_duration_minutes': stats['duration'] / 60,
            'average_posture_score': stats['average_score'],
            'good_posture_percentage': stats['good_posture_percent'],
            'good_posture_time_minutes': stats['good_posture_time'] / 60,
            'poor_posture_time_minutes': stats['poor_posture_time'] / 60,
            'feedback_count': len(self.feedback_system.feedback_history)
        }
        
        filename = f"posture_report_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Session report saved: {filename}")
        print(f"   Duration: {report['session_duration_minutes']:.1f} minutes")
        print(f"   Average Score: {report['average_posture_score']:.1f}/100")
        print(f"   Good Posture: {report['good_posture_percentage']:.1f}%")
    
    def reset_session(self):
        """Reset session statistics"""
        self.analyzer.session_start = time.time()
        self.analyzer.posture_scores = []
        self.analyzer.poor_posture_duration = 0
        self.analyzer.good_posture_duration = 0
        self.analyzer.last_score_time = time.time()
        self.feedback_system.feedback_history = []
        print("\n🔄 Session statistics reset")
    
    def cleanup(self):
        """Cleanup resources"""
        # Final statistics
        if self.calibrated:
            stats = self.analyzer.get_session_stats()
            print("\n" + "="*60)
            print("📊 FINAL SESSION STATISTICS")
            print("="*60)
            print(f"Total Duration: {stats['duration']/60:.1f} minutes")
            print(f"Average Posture Score: {stats['average_score']:.1f}/100")
            print(f"Good Posture Time: {stats['good_posture_time']/60:.1f} minutes ({stats['good_posture_percent']:.1f}%)")
            print(f"Poor Posture Time: {stats['poor_posture_time']/60:.1f} minutes")
            print(f"Total Feedback Given: {len(self.feedback_system.feedback_history)}")
            print("="*60)
            
            # Ask to save
            save = input("\n💾 Save session report? (y/n): ")
            if save.lower() == 'y':
                self.save_session_stats()
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        print("\n👋 Thank you for using Advanced Posture Advisor!")
        print("   Remember to take breaks and stretch regularly! 🧘\n")

if __name__ == "__main__":
    try:
        advisor = AdvancedPostureAdvisor()
        advisor.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()