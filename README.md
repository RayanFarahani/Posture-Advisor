# Advanced Posture Advisor using Web Camera and Pose Estimation

This project provides an intelligent posture monitoring system that uses a standard web camera together with a pose estimation model. It observes the user in real time and interprets several biomechanical cues that relate to sitting posture. The program then offers adaptive feedback that encourages sustainable alignment instead of rigid corrective behavior. One might think of it as a quiet assistant that notices patterns and reacts gently when necessary.

The system maintains a baseline posture profile during a brief calibration phase. After this initial stage, each new frame is compared to that personal baseline rather than a fixed ideal. This approach adapts to different body proportions and seating habits. The feedback mechanism is designed to reinforce improvements and discourage strain patterns without constant interruption.

The display includes posture quality metrics, a summary of deviations from baseline, and a visual overlay that highlights head and torso alignment. Feedback banners appear when posture significantly diverges from the learned reference. Over longer sessions, additional statistics are recorded, such as how much time was spent in balanced sitting.

### Key Features

• Real time pose tracking using Mediapipe  
• Personalised baseline calibration so that evaluation reflects the individual user rather than a generic model  
• Analysis of head position, spine inclination, shoulder leveling, and symmetry  
• Continuous posture scoring that adapts over time  
• Context aware feedback that reduces unnecessary interruptions  
• Visual interface panel for metrics and progress display

### Installation

A Python environment is required. Most users will find it simplest to create a separate environment before installing the dependencies.

pip install -r requirements.txt


### Running the Program

Ensure your camera is connected. Then launch the main script.

python gesture.py

At the beginning, sit naturally and remain relatively steady for a short calibration period. Once calibration finishes, the system begins monitoring automatically. Feedback appears only when a consistent deviation is detected. Sometimes minor drifting happens for a few seconds, which is normal.

### Notes

The program does not store any video feed or sensory data. Everything is processed locally. If performance issues arise, reducing the camera resolution often improves responsiveness.

Feel free to modify threshold values or add further metrics. The code is structured so that extensions can be explored without difficulty. Graduate students and hobby researchers may find this an interesting base for experiments in ergonomics or human computer interaction.

