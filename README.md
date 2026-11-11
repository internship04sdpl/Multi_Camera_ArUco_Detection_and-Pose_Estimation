Multi-Camera ArUco Detection and Pose Estimation

Project Overview

This project implements a camera-independent multi-camera vision system using OpenCV and ArUco markers.

It supports three different USB cameras and provides camera calibration, lens distortion correction, ArUco marker detection, and 3D pose estimation.

The system identifies cameras dynamically using their hardware identifiers (VID/PID), avoiding dependence on fixed camera indexes.

Supported Cameras

Camera

Camera Identifier

Sinoseen USB 2.0

sinoseen_usb2

Sinoseen USB 3.0

sinoseen_usb3

OV2710 USB 2.0

ov2710_usb2

Project Structure

Multi-Camera-ArUco-Detection-and-Pose-Estimation/
│
├── camera_intrinsics.py
├── distortion.py
├── aruco.py
├── requirements.txt
└── README.md

Technologies Used

Python

OpenCV Contrib

NumPy

ArUco Marker Detection

Camera Calibration

Pose Estimation

Python Modules

1. camera_intrinsics.py

Camera identification using VID/PID.

Dynamic camera discovery.

Camera intrinsic parameter retrieval.

Camera calibration data management.

Field of View (FOV) calculation.

2. distortion.py

Lens distortion correction.

Undistortion map generation.

Live camera distortion correction.

Original and corrected image comparison.

3. aruco.py

ArUco marker detection.

Marker ID identification.

Marker boundary visualization.

3D pose estimation using solvePnP.

Position and distance measurement.

Omega angle calculation.

Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate to the project directory:

cd Multi-Camera-ArUco-Detection-and-Pose-Estimation

Install the dependencies:

python -m pip install -r requirements.txt

Usage

List Supported Cameras

python camera_intrinsics.py list

Camera Calibration Information

python camera_intrinsics.py ov2710_usb2 info

Live Distortion Correction

python distortion.py ov2710_usb2 live

Live ArUco Detection

python aruco.py ov2710_usb2 live

Other supported cameras:

python aruco.py sinoseen_usb2 live
python aruco.py sinoseen_usb3 live

Detect All Markers

python aruco.py ov2710_usb2 all

Individual Measurements

python aruco.py ov2710_usb2 x
python aruco.py ov2710_usb2 y
python aruco.py ov2710_usb2 z_depth
python aruco.py ov2710_usb2 distance
python aruco.py ov2710_usb2 omega

ArUco Configuration

Dictionary: DICT_5X5_50

Marker Size: 41 mm

Pose Estimation: OpenCV solvePnP

Coordinate Measurements: X, Y and Z

Distance: Straight-line distance from the camera to the marker centre

Omega: Marker top-edge angle in the image plane

Measurement Outputs

The system provides:

Marker ID

X Position (mm)

Y Position (mm)

Z Depth (mm)

Straight-line Distance (mm)

Omega Angle (degrees)

Rotation Vector (rvec)

Translation Vector (tvec)

Notes

Camera calibration data must be available for the selected camera.

Connect the required camera before running the program.

Use the matching ArUco dictionary and marker size.

Press Q to close live camera windows.

Author

Developed as a multi-camera computer vision project.
