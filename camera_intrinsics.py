
"""
camera_intrinsics.py

Reusable camera intrinsics and calibration library with dynamic USB camera discovery.

Supported cameras:
1. sinoseen_usb2
2. sinoseen_usb3
3. ov2710_usb2

Commands:
    python camera_intrinsics.py list
    python camera_intrinsics.py sinoseen_usb2 fx
    python camera_intrinsics.py sinoseen_usb2 matrix
    python camera_intrinsics.py sinoseen_usb2 distortion
    python camera_intrinsics.py sinoseen_usb2 fov
    python camera_intrinsics.py sinoseen_usb2 all
    python camera_intrinsics.py sinoseen_usb2 calibrate
"""

import cv2
import numpy as np
import math
import argparse
import time

try:
    from cv2_enumerate_cameras import enumerate_cameras
except ImportError:
    enumerate_cameras = None


# =========================================================
# 1. CAMERA CONFIGURATION
# =========================================================

CAMERAS = {
    "sinoseen_usb2": {
        "name": "Sinoseen USB 2.0",
        "vid": "2304",
        "pid": "1689",
        "width": 640,
        "height": 480,
    },
    "sinoseen_usb3": {
        "name": "Sinoseen USB 3.0",
        "vid": "1BCF",
        "pid": "2D52",
        "width": 640,
        "height": 480,
    },
    "ov2710_usb2": {
        "name": "OV2710 USB 2.0",
        "vid": "0C45",
        "pid": "6369",
        "width": 640,
        "height": 480,
    },
}


# =========================================================
# 2. EXISTING CALIBRATION VALUES
# =========================================================

CALIBRATIONS = {
    "sinoseen_usb2": {
        "camera_matrix": np.array([
            [438.01763699, 0.0, 327.68438151],
            [0.0, 445.36487274, 267.70436656],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64),

        "distortion_coefficients": np.array([
            -0.42687719,
            0.36299048,
            -0.00530278,
            -0.00128296,
            -0.18549303,
        ], dtype=np.float64),
    },

    "sinoseen_usb3": {
        "camera_matrix": np.array([
            [327.79489842, 0.0, 307.23325896],
            [0.0, 328.35131594, 226.74646758],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64),

        "distortion_coefficients": np.array([
            0.07293927,
            0.02675937,
            -0.00129332,
            -0.00601109,
            -0.06795796,
        ], dtype=np.float64),
    },

    # Calibration must be performed for this camera.
    
    "ov2710_usb2": {
        "camera_matrix": np.array([
            [427.9100351831, 0.0, 357.2505034285],
            [0.0, 430.1109723004, 266.9827000369],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64),

        "distortion_coefficients": np.array([
            -0.3577505085,
            0.1135886045,
            -0.0097617099,
            -0.0001867036,
            0.0019177494,
        ], dtype=np.float64),
    },

}


# =========================================================
# 3. CHESSBOARD SETTINGS
# =========================================================

CHESSBOARD_INNER_CORNERS = (11, 11)
CHESSBOARD_SQUARE_SIZE_MM = 10.0
TARGET_CALIBRATION_VIEWS = 25
MIN_CALIBRATION_VIEWS = 10


# =========================================================
# 4. CAMERA INFORMATION FUNCTIONS
# =========================================================

def list_cameras():
    """Return the supported camera names (not the currently connected list)."""
    return list(CAMERAS.keys())


def get_camera_info(camera_name):
    """Return the saved identity and image settings for a supported camera."""
    name = camera_name.lower().strip()

    if name not in CAMERAS:
        raise ValueError(
            f"Unknown camera: {camera_name}. "
            f"Available cameras: {', '.join(CAMERAS)}"
        )

    return CAMERAS[name].copy()


def _format_usb_id(value):
    """Normalize the numeric VID/PID returned by cv2-enumerate-cameras."""
    if value is None:
        return None
    return f"{int(value):04X}"


def discover_camera(camera_name):
    """Find the requested physical camera and return its current DSHOW index.

    The index is discovered every time this function runs. It is never stored
    in CAMERAS and may change whenever USB devices are connected or removed.
    """
    name = camera_name.lower().strip()
    info = get_camera_info(name)

    if enumerate_cameras is None:
        raise RuntimeError(
            "Camera discovery package is missing. Install it with: "
            "python -m pip install cv2-enumerate-cameras"
        )

    try:
        devices = list(enumerate_cameras(cv2.CAP_DSHOW))
    except Exception as error:
        raise RuntimeError(f"Could not enumerate Windows cameras: {error}") from error

    expected_vid = info["vid"].upper()
    expected_pid = info["pid"].upper()
    matches = []

    for device in devices:
        device_vid = _format_usb_id(getattr(device, "vid", None))
        device_pid = _format_usb_id(getattr(device, "pid", None))

        if device_vid == expected_vid and device_pid == expected_pid:
            matches.append({
                "index": int(device.index),
                "name": str(device.name),
                "vid": device_vid,
                "pid": device_pid,
                "path": str(getattr(device, "path", "")),
            })

    if not matches:
        raise RuntimeError(
            f"{info['name']} is not currently detected by Windows. "
            f"Expected USB VID:PID {expected_vid}:{expected_pid}. "
            "Check its USB connection and try again."
        )

    if len(matches) > 1:
        paths = "\n".join(item["path"] for item in matches)
        raise RuntimeError(
            f"More than one connected device matches {info['name']} "
            f"(VID:PID {expected_vid}:{expected_pid}). Cannot safely choose. "
            f"Matching device paths:\n{paths}"
        )

    return matches[0]


def get_camera_index(camera_name):
    """Return the camera's currently discovered index (diagnostics only)."""
    return discover_camera(camera_name)["index"]


# =========================================================
# 5. CALIBRATION ACCESS FUNCTIONS
# =========================================================

def load_calibration(camera_name):
    """Return copies of the camera matrix and distortion coefficients."""
    name = camera_name.lower().strip()

    if name not in CAMERAS:
        raise ValueError(f"Unknown camera: {camera_name}")

    calibration = CALIBRATIONS.get(name)

    if calibration is None:
        raise RuntimeError(
            f"Calibration is not available for {name}. "
            "Run the calibrate command and add its results to "
            "the CALIBRATIONS dictionary."
        )

    return (
        calibration["camera_matrix"].copy(),
        calibration["distortion_coefficients"].copy(),
    )


def get_matrix(camera_name):
    """Return the 3 x 3 intrinsic camera matrix."""
    matrix, _ = load_calibration(camera_name)
    return matrix


def get_distortion_coefficients(camera_name):
    """Return the camera's distortion coefficients."""
    _, coefficients = load_calibration(camera_name)
    return coefficients


# =========================================================
# 6. INTRINSIC VALUES AND FOV
# =========================================================

def get_value(camera_name, value_name):
    """
    Supported:
        fx, fy, cx, cy
        matrix, camera_matrix
        hfov, vfov, dfov
        distortion, coefficients, k1, k2, p1, p2, k3
    """
    name = value_name.lower().strip()

    matrix, coefficients = load_calibration(camera_name)
    info = get_camera_info(camera_name)

    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])

    width = info["width"]
    height = info["height"]

    coefficient_names = ["k1", "k2", "p1", "p2", "k3"]
    coeffs = coefficients.flatten()

    values = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "matrix": matrix.tolist(),
        "camera_matrix": matrix.tolist(),

        "distortion": coeffs.tolist(),
        "coefficients": coeffs.tolist(),
        "distortion_coefficients": coeffs.tolist(),

        "hfov": math.degrees(
            2.0 * math.atan(width / (2.0 * fx))
        ),
        "vfov": math.degrees(
            2.0 * math.atan(height / (2.0 * fy))
        ),
        "dfov": math.degrees(
            2.0 * math.atan(
                math.sqrt(width ** 2 + height ** 2)
                / (2.0 * math.sqrt(fx ** 2 + fy ** 2))
            )
        ),
    }

    for index, key in enumerate(coefficient_names):
        if index < len(coeffs):
            values[key] = float(coeffs[index])

    if name not in values:
        raise ValueError(
            "Supported values: fx, fy, cx, cy, matrix, camera_matrix, "
            "hfov, vfov, dfov, distortion, coefficients, "
            "distortion_coefficients, k1, k2, p1, p2, k3."
        )

    return values[name]


def get_fov(camera_name):
    """Return horizontal, vertical, and diagonal FOV in degrees."""
    return {
        "hfov": get_value(camera_name, "hfov"),
        "vfov": get_value(camera_name, "vfov"),
        "dfov": get_value(camera_name, "dfov"),
    }


# =========================================================
# 7. PRINT ALL AVAILABLE CALIBRATION VALUES
# =========================================================

def print_all(camera_name):
    """Print all stored intrinsic and distortion values."""
    info = get_camera_info(camera_name)

    print("\n" + "=" * 60)
    print("CAMERA INTRINSICS REPORT")
    print("=" * 60)

    print("Camera name:", info["name"])
    print("Camera key:", camera_name)
    print("USB identity:", f"VID:PID {info['vid']}:{info['pid']}")
    print(
        "Configured resolution:",
        f"{info['width']} x {info['height']}"
    )

    matrix, coefficients = load_calibration(camera_name)

    print("\nCamera matrix:")
    print(matrix)

    print("\nIndividual intrinsic values:")
    print(f"fx: {matrix[0, 0]:.10f}")
    print(f"fy: {matrix[1, 1]:.10f}")
    print(f"cx: {matrix[0, 2]:.10f}")
    print(f"cy: {matrix[1, 2]:.10f}")

    print("\nDistortion coefficients:")
    for key, value in zip(
        ["k1", "k2", "p1", "p2", "k3"],
        coefficients.flatten()
    ):
        print(f"{key}: {value:.10f}")

    print("\nField of view:")
    fov = get_fov(camera_name)
    print(f"Horizontal FOV: {fov['hfov']:.4f} degrees")
    print(f"Vertical FOV:   {fov['vfov']:.4f} degrees")
    print(f"Diagonal FOV:   {fov['dfov']:.4f} degrees")

    print("=" * 60)


# =========================================================
# 8. OPEN CAMERA
# =========================================================

def open_camera(camera_name):
    """Discover and open the requested camera without a fixed index."""
    info = get_camera_info(camera_name)
    device = discover_camera(camera_name)

    cap = cv2.VideoCapture(device["index"], cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Windows found {info['name']} (VID:PID "
            f"{info['vid']}:{info['pid']}) at its current device index, "
            "but OpenCV could not open it. Close other camera applications "
            "and try again."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, info["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, info["height"])

    time.sleep(0.5)
    return cap


# =========================================================
# 9. CHESSBOARD CALIBRATION
# =========================================================

def calibrate_camera(camera_name):
    """
    Capture varied chessboard views and calculate calibration.
    The results are printed so they can be copied into CALIBRATIONS.
    """
    name = camera_name.lower().strip()
    info = get_camera_info(name)

    pattern_size = CHESSBOARD_INNER_CORNERS
    square_size = CHESSBOARD_SQUARE_SIZE_MM

    object_template = np.zeros(
        (pattern_size[0] * pattern_size[1], 3),
        dtype=np.float32,
    )

    object_template[:, :2] = np.mgrid[
        0:pattern_size[0],
        0:pattern_size[1],
    ].T.reshape(-1, 2)

    object_template *= square_size

    object_points = []
    image_points = []
    image_size = None

    cap = open_camera(name)

    last_center = None
    last_area = None
    last_angle = None
    last_capture_time = 0.0

    print("\nCamera calibration started")
    print("Camera:", info["name"])
    print("Resolution:", info["width"], "x", info["height"])
    print("Inner corners:", pattern_size)
    print("Square size:", square_size, "mm")
    print("Target views:", TARGET_CALIBRATION_VIEWS)
    print("Move the board to different positions and angles.")
    print("Press Q to finish; press ESC to cancel.")

    try:
        while len(object_points) < TARGET_CALIBRATION_VIEWS:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Could not read a camera frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])

            found, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                flags=(
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    | cv2.CALIB_CB_NORMALIZE_IMAGE
                ),
            )

            display = frame.copy()
            status = "Chessboard not detected"

            if found:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (
                        cv2.TERM_CRITERIA_EPS
                        + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001,
                    ),
                )

                cv2.drawChessboardCorners(
                    display,
                    pattern_size,
                    corners,
                    found,
                )

                pts = corners.reshape(-1, 2)
                center = pts.mean(axis=0)

                area = float(
                    (pts[:, 0].max() - pts[:, 0].min())
                    * (pts[:, 1].max() - pts[:, 1].min())
                )

                dx = float(
                    pts[pattern_size[0] - 1, 0] - pts[0, 0]
                )
                dy = float(
                    pts[pattern_size[0] - 1, 1] - pts[0, 1]
                )
                angle = math.degrees(math.atan2(dy, dx))
                now = time.monotonic()

                should_capture = last_center is None

                if (
                    last_center is not None
                    and now - last_capture_time >= 0.7
                ):
                    center_change = float(
                        np.linalg.norm(center - last_center)
                    )
                    area_change = (
                        abs(area - last_area) / max(last_area, 1.0)
                    )
                    angle_change = abs(
                        (angle - last_angle + 180.0) % 360.0 - 180.0
                    )

                    should_capture = (
                        center_change >= 25.0
                        or area_change >= 0.12
                        or angle_change >= 10.0
                    )

                if should_capture:
                    object_points.append(object_template.copy())
                    image_points.append(corners.copy())

                    last_center = center.copy()
                    last_area = area
                    last_angle = angle
                    last_capture_time = now

                    print(
                        f"Captured view {len(object_points)}/"
                        f"{TARGET_CALIBRATION_VIEWS}"
                    )

                status = (
                    f"Captured views: {len(object_points)}/"
                    f"{TARGET_CALIBRATION_VIEWS}"
                )

            cv2.putText(
                display,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )

            cv2.imshow(f"Calibration - {info['name']}", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key == 27:
                print("Calibration cancelled.")
                return None

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(object_points) < MIN_CALIBRATION_VIEWS:
        raise RuntimeError(
            f"Only {len(object_points)} varied views were captured. "
            f"At least {MIN_CALIBRATION_VIEWS} are required."
        )

    print("\nCalculating calibration...")

    rms, matrix, coefficients, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    total_error = 0.0

    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            matrix,
            coefficients,
        )

        detected = image_points[i].reshape(-1, 2)
        projected = projected.reshape(-1, 2)

        error = cv2.norm(
            detected.astype(np.float32),
            projected.astype(np.float32),
            cv2.NORM_L2,
        ) / len(projected)

        total_error += error

    mean_error = total_error / len(object_points)

    print("\n========== CALIBRATION RESULTS ==========")
    print("Camera:", name)
    print("Views used:", len(object_points))
    print("RMS reprojection error:", float(rms))
    print("Mean reprojection error:", float(mean_error), "pixels")
    print("\nCamera matrix:")
    print(matrix)
    print("\nDistortion coefficients:")
    print(coefficients.ravel())

    print("\n========== COPY INTO CALIBRATIONS ==========")
    print(f'"{name}": {{')
    print('    "camera_matrix": np.array([')

    for row in matrix:
        print(
            "        ["
            + ", ".join(f"{value:.10f}" for value in row)
            + "],"
        )

    print("    ], dtype=np.float64),")
    print('    "distortion_coefficients": np.array([')
    print(
        "        "
        + ", ".join(
            f"{value:.10f}" for value in coefficients.ravel()
        )
    )
    print("    ], dtype=np.float64),")
    print("},")

    print("\nCopy the printed entry into CALIBRATIONS.")
    print("Do not replace another camera's calibration accidentally.")

    return {
        "camera_matrix": matrix,
        "distortion_coefficients": coefficients.ravel(),
        "rms": float(rms),
        "mean_reprojection_error": float(mean_error),
        "views": len(object_points),
    }


# =========================================================
# 10. COMMAND-LINE INTERFACE
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reusable camera intrinsics and calibration library"
    )

    parser.add_argument(
        "camera",
        nargs="?",
        help="Camera name or 'list'",
    )

    parser.add_argument(
        "value",
        nargs="?",
        help="fx, fy, cx, cy, matrix, distortion, fov, all, "
             "info, index, hfov, vfov, dfov, or calibrate",
    )

    args = parser.parse_args()

    try:
        if args.camera is None or args.camera.lower() == "list":
            print("Supported cameras:")
            for name, info in CAMERAS.items():
                print(
                    f"  {name}: {info['name']} "
                    f"(VID:PID {info['vid']}:{info['pid']}, "
                    f"{info['width']}x{info['height']})"
                )
            return

        camera_name = args.camera.lower().strip()

        if camera_name not in CAMERAS:
            parser.error(
                f"Unknown camera '{camera_name}'. "
                f"Available: {', '.join(CAMERAS)}"
            )

        if args.value is None:
            parser.error("Specify a value, for example fx, matrix, or all.")

        value = args.value.lower().strip()

        if value == "info":
            print("Camera information:", get_camera_info(camera_name))
            print("Currently discovered device:", discover_camera(camera_name))

        elif value == "index":
            # Dynamic diagnostic value; it is not saved in the configuration.
            print(get_camera_index(camera_name))

        elif value == "calibrate":
            calibrate_camera(camera_name)

        elif value == "all":
            print_all(camera_name)

        elif value == "fov":
            print(get_fov(camera_name))

        elif value == "distortion":
            print(get_distortion_coefficients(camera_name))

        else:
            result = get_value(camera_name, value)
            print(f"Camera: {CAMERAS[camera_name]['name']}")
            print(f"Value: {value}")
            print(f"Result: {result}")

    except (ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
