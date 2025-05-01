# AI-Driven Microexpression-Based Lie Detection System

This repository contains an AICTE Capstone project that implements a real-time system for detecting microexpressions and assessing potential deception using deep learning. The system analyzes facial expressions from video input, predicts emotions, and calculates a deception score to determine if a person might be lying.

## Features
- Real-time face detection using OpenCV and DeepFace.
- Emotion analysis with DeepFace and a custom CNN+RNN model (PyTorch).
- Deception score calculation based on temporal microexpression patterns.
- Output visualization with labeled frames and emotion probability plots.

## Technologies Used
- **Python 3.12.5**
- **PyTorch 2.3.1+cpu**
- **OpenCV**
- **DeepFace 0.0.93**
- **Matplotlib** (for plotting)
- **SciPy** (for smoothing deception scores)

## Installation

### Prerequisites
- Python 3.12 or later
- Git (for cloning the repository)
