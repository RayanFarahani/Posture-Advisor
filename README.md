# Advanced Posture Advisor using Web Camera and Pose Estimation

Real-time posture monitoring using a standard webcam + **MediaPipe Pose**. The app calibrates to *your* comfortable baseline posture, then provides gentle feedback when you deviate for a sustained period (instead of spamming alerts).

The system maintains a baseline posture profile during a brief calibration phase. After this initial stage, each new frame is compared to that personal baseline rather than a fixed ideal. This approach adapts to different body proportions and seating habits. The feedback mechanism is designed to reinforce improvements and discourage strain patterns without constant interruption.

The display includes posture quality metrics, a summary of deviations from baseline, and a visual overlay that highlights head and torso alignment. Feedback banners appear when posture significantly diverges from the learned reference. Over longer sessions, additional statistics are recorded, such as how much time was spent in balanced sitting.

## Key features

- **Real-time pose tracking** with MediaPipe
- **Personalized baseline calibration** (evaluates relative to you, not a generic “ideal”)
- **Metrics**: head forward posture, spine inclination, shoulder leveling, symmetry, confidence
- **Adaptive posture score** (0–100) with persistent-issue detection
- **Context-aware feedback** with cooldown to reduce interruptions
- **Session stats** export to JSON (optional)

## Installation

A Python environment is required. Most users will find it simplest to create a separate environment before installing the dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python gesture.py
```

At the beginning, sit naturally and remain relatively steady for a short calibration period. Once calibration finishes, the system begins monitoring automatically. Feedback appears only when a consistent deviation is detected (minor drifting for a few seconds is normal).

## Controls

- **Space**: calibrate / recalibrate baseline posture
- **P**: pause/resume monitoring
- **S**: save session statistics
- **R**: reset session statistics
- **Q**: quit

## Notes

- The program does **not** store video frames; processing is local.
- If performance is slow, reducing camera resolution can help.

Feel free to modify threshold values or add further metrics. The code is structured so that extensions can be explored without difficulty. Graduate students and hobby researchers may find this an interesting base for experiments in ergonomics or human computer interaction.

