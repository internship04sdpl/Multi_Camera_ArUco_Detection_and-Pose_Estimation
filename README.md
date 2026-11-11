# Multi-Camera ArUco Detection and Pose Estimation

A Python-based computer vision project for multi-camera calibration, lens distortion correction, ArUco marker detection, and pose estimation using OpenCV.

The system supports three USB cameras and identifies them dynamically using their hardware Vendor ID (VID) and Product ID (PID), rather than relying on fixed camera indexes.

## Project Overview

This project consists of three Python modules:

1. **Camera Intrinsics** – Camera identification, calibration data, intrinsic parameters, and Field of View (FOV).
2. **Distortion Correction** – Lens distortion correction and live undistorted camera output.
3. **ArUco Detection** – Marker detection, ID identification, position estimation, and distance measurement.

## Supported Cameras

| No. | Camera | Camera Identifier |
|---|---|---|
| 1 | Sinoseen USB 2.0 | `sinoseen_usb2` |
| 2 | Sinoseen USB 3.0 | `sinoseen_usb3` |
| 3 | OV2710 USB 2.0 | `ov2710_usb2` |

## Project Structure

```text
Multi_Camera_ArUco_Detection_and_Pose_Estimation/
│
├── camera_intrinsics.py
├── distortion.py
├── aruco.py
├── requirements.txt
└── README.md
```

## Technologies and Libraries

- Python
- OpenCV Contrib
- NumPy
- ArUco
- Camera Calibration
- Computer Vision
- Pose Estimation

## Python Modules

### 1. camera_intrinsics.py

Responsible for camera identification and calibration.

Features:
- Dynamic camera discovery.
- Camera identification using VID and PID.
- Camera intrinsic matrix retrieval.
- Distortion coefficient retrieval.
- Camera calibration management.
- Horizontal, vertical, and diagonal FOV calculation.

### 2. distortion.py

Responsible for correcting camera lens distortion.

Features:
- Lens distortion correction.
- Undistortion map generation.
- Frame remapping.
- Live distortion correction.
- Original and corrected image comparison.

### 3. aruco.py

Responsible for ArUco marker detection and pose estimation.

Features:
- ArUco marker detection.
- Marker ID identification.
- Marker boundary visualization.
- 3D position estimation using OpenCV `solvePnP`.
- X, Y, and Z-depth measurements.
- Straight-line distance calculation.
- Omega angle calculation.
- Rotation vector (rvec) and translation vector (tvec).

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/internship04sdpl/Multi_Camera_ArUco_Detection_and_Pose_Estimation.git
```

### 2. Navigate to the Project Directory

```bash
cd Multi_Camera_ArUco_Detection_and_Pose_Estimation
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Usage

### 1. List Supported Cameras

```bash
python aruco.py list
```

### 2. Live Camera Distortion Correction

**Sinoseen USB 2.0**

```bash
python distortion.py sinoseen_usb2 live
```

**Sinoseen USB 3.0**

```bash
python distortion.py sinoseen_usb3 live
```

**OV2710 USB 2.0**

```bash
python distortion.py ov2710_usb2 live
```

### 3. Live ArUco Detection

**Sinoseen USB 2.0**

```bash
python aruco.py sinoseen_usb2 live
```

**Sinoseen USB 3.0**

```bash
python aruco.py sinoseen_usb3 live
```

**OV2710 USB 2.0**

```bash
python aruco.py ov2710_usb2 live
```

Press `Q` to close the live camera window.

### 4. Detect All ArUco Markers

```bash
python aruco.py ov2710_usb2 all
```

### 5. Detect a Specific Marker ID

```bash
python aruco.py ov2710_usb2 live --marker-id 4
```

## ArUco Configuration

| Parameter | Configuration |
|---|---|
| Dictionary | `DICT_5X5_50` |
| Marker Size | 41 mm |
| Pose Estimation | `cv2.solvePnP()` |
| Pose Estimation Method | `SOLVEPNP_IPPE_SQUARE` |
| Detection | Multiple marker IDs |

## Measurement Outputs

The system provides the following measurements:

| Parameter | Description | Unit |
|---|---|---|
| ID | Detected ArUco marker ID | Number |
| X | Horizontal position relative to camera | mm |
| Y | Vertical position relative to camera | mm |
| Z-depth | Depth from camera to marker centre | mm |
| Distance | Straight-line distance to marker centre | mm |
| Omega | Marker top-edge angle in the image plane | Degrees |
| rvec | Rotation vector | — |
| tvec | Translation vector | — |

## Camera Calibration

Each supported camera uses its own calibration data.

The calibration parameters include:
- Camera intrinsic matrix.
- Distortion coefficients.
- Focal lengths.
- Optical centre coordinates.
- Field of View (FOV).

**Note:** Calibration data must be available for the selected camera before running distortion correction or pose estimation.

## Requirements

Dependencies are listed in `requirements.txt`.

Install them using:

```bash
python -m pip install -r requirements.txt
```

## Notes

- Connect the required camera before starting detection.
- Camera selection is based on the configured camera identity.
- Use the correct ArUco dictionary and physical marker size.
- Ensure the marker is visible and properly focused.
- X, Y, and Z measurements are expressed in millimetres.
- Omega represents the 2D image-plane angle, not the complete 3D orientation.

## Repository

**GitHub:** https://github.com/internship04sdpl/Multi_Camera_ArUco_Detection_and_Pose_Estimation

