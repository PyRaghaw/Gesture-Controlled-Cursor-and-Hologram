# 🦾 Iron Man Hologram Controller

An advanced gesture-controlled hologram interface inspired by Tony Stark's lab, enabling touchless 3D object manipulation using hand tracking.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Problem Statement

Traditional CAD software and 3D modeling tools require extensive mouse/keyboard interaction, creating barriers in:
- **EdTech**: Students struggle to engage with 3D geometry concepts
- **MedTech**: Surgeons need sterile, touchless interfaces for viewing medical imaging
- **Architecture**: Designers need intuitive ways to present and manipulate 3D models in client meetings

This project solves these challenges by enabling natural, gesture-based interaction with 3D objects—no physical contact required.

## ✨ Key Features

### 🎮 Hologram Mode
- **8 Pre-built 3D Models**: Cube, Sphere, Pyramid, Torus, Cylinder, Diamond, Helix, Star
- **Multi-object Support**: Spawn and manipulate multiple objects simultaneously
- **Gesture Controls**:
  - 👌 **Pinch**: Select objects from menu
  - 👆 **Index Finger**: Drag & position objects
  - 🖐️ **Open Palm**: Rotate objects in 3D space
  - 🤲 **Two Hands**: Pinch-to-zoom (spread to enlarge, pinch to shrink)
  - 🗑️ **Delete Zone**: Drag objects to top-right corner to delete

### 💻 Desktop Control (30+ Gestures)
- Cursor movement, left/right click, double-click
- Scrolling, volume control, brightness adjustment
- App switching, screenshot capture, media playback
- Presentation mode with slide navigation
- Drawing mode with virtual canvas

### 🎨 Additional Modes
- **Drawing Mode**: Air-draw with finger tracking
- **Gaming Mode**: Gesture-mapped WASD controls
- **Presentation Mode**: Navigate slides with swipe gestures

## 🛠️ Tech Stack

- **Computer Vision**: OpenCV 4.x
- **Hand Tracking**: MediaPipe Hands (Google)
- **Desktop Automation**: PyAutoGUI
- **3D Rendering**: Custom projection engine with perspective transformation
- **Language**: Python 3.8+

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/hologram-controller.git
cd hologram-controller

# Install dependencies
pip install opencv-python mediapipe pyautogui numpy

# Run the application
python hologram_controller.py
```

## 🎬 Usage

### Quick Start
1. Run `python hologram_controller.py`
2. Press **H** to enter Hologram Mode
3. Use gestures to interact with 3D objects

### Keyboard Controls
- **H**: Toggle Hologram Mode
- **M**: Cycle through modes
- **D**: Drawing Mode
- **P**: Presentation Mode
- **C**: Clear canvas/objects
- **X**: Deselect all objects
- **S**: Show statistics
- **Q**: Quit

### Gesture Guide (Hologram Mode)
| Gesture | Fingers | Action |
|---------|---------|--------|
| Pinch | Thumb + Index | Select/Spawn object |
| Point | Index only | Drag object |
| Open Palm | All 5 | Rotate object |
| Two Hands | Both palms | Zoom in/out |

## 🏥 Real-World Applications

### EdTech
- Interactive geometry lessons
- 3D molecule visualization for chemistry
- Hands-on STEM learning without expensive equipment

### MedTech
- Sterile medical imaging manipulation in operating rooms
- Touchless anatomy exploration for training
- Patient scan review during consultations

### Architecture
- Client presentations with live 3D model interaction
- Collaborative design reviews
- Site planning visualization

## 🧠 How It Works

1. **Hand Detection**: MediaPipe tracks 21 hand landmarks in real-time
2. **Gesture Recognition**: Custom algorithm analyzes finger positions and movements
3. **3D Projection**: Wireframe objects rendered with perspective transformation
4. **Action Execution**: Gestures mapped to object manipulation functions

## 📊 Performance

- **FPS**: 25-30 on standard webcams
- **Latency**: <50ms gesture recognition
- **Accuracy**: 90%+ detection confidence (configurable)

## 🔮 Future Enhancements

- [ ] Voice command integration
- [ ] AR headset support
- [ ] Collaborative multi-user mode
- [ ] Custom object import (OBJ/STL files)
- [ ] Haptic feedback via wearables
- [ ] Cloud-based model sharing

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR for:
- New 3D models
- Additional gesture patterns
- Performance optimizations
- Bug fixes

## 📄 License

MIT License - feel free to use this project for educational or commercial purposes.

## 🙏 Acknowledgments

- MediaPipe by Google Research
- OpenCV Community
- Inspired by Iron Man's holographic interface

## 📧 Contact

For questions or collaborations:
- LinkedIn: Raghaw Shukla(https://www.linkedin.com/in/raghaw-shukla-a49727326/)

---

**Star ⭐ this repo if you found it helpful!**
