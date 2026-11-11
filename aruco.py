"""
aruco.py

ArUco detection and pose estimation using:
    camera_intrinsics.py
    distortion.py

Supported cameras:
    sinoseen_usb2
    sinoseen_usb3
    ov2710_usb2

Measurements:
    ID, X, Y, Z-depth, straight-line distance, Omega,
    rotation vector (rvec), translation vector (tvec)

Default marker size: 41 mm
Dictionary: DICT_5X5_50

Examples:
    python aruco.py list
    python aruco.py sinoseen_usb3 all
    python aruco.py sinoseen_usb3 live
    python aruco.py sinoseen_usb3 x
    python aruco.py sinoseen_usb3 all --attempts 100
"""

import argparse
import math

import cv2
import numpy as np

import camera_intrinsics as ci
import distortion as dist


# ============================================================
# SETTINGS
# ============================================================

MARKER_SIZE_MM = 41.0
DICTIONARY = cv2.aruco.DICT_5X5_50

WINDOW_NAME = "ArUco Detection"

# Marker boundary colour: ORANGE (BGR)
BOUNDARY_COLOUR = (0, 165, 255)

# Marker ID colour: BLUE (BGR)
ID_COLOUR = (255, 0, 0)

# Measurement text colour: WHITE
TEXT_COLOUR = (0, 255, 0)

# Text outline colour: BLACK
TEXT_OUTLINE = (0, 0, 0)

# Measurement position
TEXT_X = 10
TEXT_START_Y = 22
TEXT_SPACING = 23

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1


# ============================================================
# CREATE ARUCO DETECTOR
# ============================================================

def create_detector():

    dictionary = cv2.aruco.getPredefinedDictionary(
        DICTIONARY
    )

    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):

        detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters
        )

        return detector, dictionary, parameters

    return None, dictionary, parameters


# ============================================================
# DETECT MARKER CORNERS
# ============================================================

def detect_corners(gray, detector_bundle):

    detector, dictionary, parameters = detector_bundle

    if detector is not None:

        corners, ids, rejected = detector.detectMarkers(gray)

    else:

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters
        )

    return corners, ids, rejected


# ============================================================
# MARKER OBJECT POINTS
# ============================================================

def get_object_points(marker_size_mm=MARKER_SIZE_MM):

    if marker_size_mm <= 0:
        raise ValueError(
            "Marker size must be greater than zero."
        )

    half = float(marker_size_mm) / 2.0

    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32
    )


# ============================================================
# CALCULATE OMEGA
# ============================================================

def calculate_omega(image_points):

    """
    Calculate the marker's top-edge angle in the image plane.

    This is a 2D image angle, not the complete 3D rotation.
    """

    dx = float(
        image_points[1, 0] - image_points[0, 0]
    )

    dy = float(
        image_points[1, 1] - image_points[0, 1]
    )

    return math.degrees(
        math.atan2(dy, dx)
    ) % 360.0


# ============================================================
# GET CAMERA MATRIX
# ============================================================

def get_pose_camera_matrix(camera_name, width, height):

    """
    Get the camera matrix corresponding to the
    undistorted image.
    """

    _, _, new_matrix, _ = dist.create_maps(
        camera_name,
        width,
        height
    )

    return new_matrix


# ============================================================
# ESTIMATE MARKER POSE
# ============================================================

def estimate_marker_poses(
    corrected_frame,
    camera_name,
    detector_bundle,
    marker_size_mm=MARKER_SIZE_MM,
    marker_id=None
):

    height, width = corrected_frame.shape[:2]

    camera_matrix = get_pose_camera_matrix(
        camera_name,
        width,
        height
    )

    gray = cv2.cvtColor(
        corrected_frame,
        cv2.COLOR_BGR2GRAY
    )

    corners, ids, _ = detect_corners(
        gray,
        detector_bundle
    )

    if ids is None:
        return []

    object_points = get_object_points(
        marker_size_mm
    )

    results = []

    for index, detected_id in enumerate(ids.flatten()):

        detected_id = int(detected_id)

        # Optional marker ID filter
        if marker_id is not None:

            if detected_id != int(marker_id):
                continue

        image_points = corners[index].reshape(
            4, 2
        ).astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            continue

        # X, Y, Z position in millimetres
        x = float(tvec[0, 0])
        y = float(tvec[1, 0])
        z_depth = float(tvec[2, 0])

        # Straight-line distance
        distance = math.sqrt(
            x * x +
            y * y +
            z_depth * z_depth
        )

        # Omega angle
        omega = calculate_omega(
            image_points
        )

        results.append(
            {
                "id": detected_id,
                "x": x,
                "y": y,
                "z_depth": z_depth,
                "distance": distance,
                "omega": omega,
                "rvec": rvec.flatten().tolist(),
                "tvec": tvec.flatten().tolist(),
                "corners": image_points.tolist()
            }
        )

    return results


# ============================================================
# DETECT ALL MARKERS IN ONE FRAME
# ============================================================

def detect(
    frame,
    camera_name,
    marker_size_mm=MARKER_SIZE_MM
):

    if frame is None:
        raise ValueError(
            "Input frame is None."
        )

    camera_name = camera_name.lower().strip()

    ci.get_camera_info(camera_name)
    ci.load_calibration(camera_name)

    corrected = dist.undistort_frame(
        frame,
        camera_name
    )

    return estimate_marker_poses(
        corrected,
        camera_name,
        create_detector(),
        marker_size_mm=marker_size_mm
    )


# ============================================================
# OPEN CAMERA
# ============================================================

def open_capture(camera_name):

    """
    Discover and open the requested camera dynamically.
    No camera index is hardcoded here.
    """

    info = ci.get_camera_info(camera_name)

    cap = ci.open_camera(camera_name)

    if not cap.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not open camera: {camera_name}"
        )

    frame = None

    # Retry camera reads
    for _ in range(10):

        success, frame = cap.read()

        if success and frame is not None:
            return cap, frame

        cv2.waitKey(50)

    cap.release()

    raise RuntimeError(
        f"Camera {camera_name} opened but did not provide a frame."
    )


# ============================================================
# DRAW MARKER BOUNDARY AND ID
# ============================================================

def draw_marker(frame, item):

    points = np.asarray(
        item["corners"],
        dtype=np.int32
    )

    # Draw ORANGE marker boundary
    cv2.polylines(
        frame,
        [points],
        True,
        BOUNDARY_COLOUR,
        2,
        cv2.LINE_AA
    )

    # Marker centre
    center = points.mean(
        axis=0
    ).astype(int)

    # Draw BLUE marker ID
    cv2.putText(
        frame,
        f"ID={item['id']}",
        (
            int(center[0]),
            int(center[1])
        ),
        FONT,
        0.6,
        ID_COLOUR,
        2,
        cv2.LINE_AA
    )


# ============================================================
# DRAW VERTICAL MEASUREMENTS
# ============================================================

def draw_measurements(frame, results):

    """
    Display measurements vertically at the top-left
    of the camera frame.

    No separate panel.
    Same position as the original screenshot.
    """

    y_position = TEXT_START_Y

    if not results:

        lines = [
            "No marker detected"
        ]

    else:

        lines = []

        for item in results:

            lines.append(
                f"ID: {item['id']}"
            )

            lines.append(
                f"X: {item['x']:.1f} mm"
            )

            lines.append(
                f"Y: {item['y']:.1f} mm"
            )

            lines.append(
                f"Depth: {item['z_depth']:.1f} mm"
            )

            lines.append(
                f"Distance: {item['distance']:.1f} mm"
            )

            lines.append(
                f"Omega: {item['omega']:.1f} deg"
            )

            lines.append("")

    for line in lines:

        if line == "":
            y_position += 10
            continue

        # Black outline for readability
        cv2.putText(
            frame,
            line,
            (TEXT_X, y_position),
            FONT,
            FONT_SCALE,
            TEXT_OUTLINE,
            3,
            cv2.LINE_AA
        )

        # White measurement text
        cv2.putText(
            frame,
            line,
            (TEXT_X, y_position),
            FONT,
            FONT_SCALE,
            TEXT_COLOUR,
            FONT_THICKNESS,
            cv2.LINE_AA
        )

        y_position += TEXT_SPACING

    return frame


# ============================================================
# GET SINGLE MEASUREMENT
# ============================================================

def get_value(
    camera_name,
    value_name,
    marker_id=None,
    attempts=30
):

    camera_name = camera_name.lower().strip()

    name = value_name.lower().strip()

    aliases = {
        "marker_id": "id",
        "z_distance": "distance",
        "omega_angle": "omega",
        "rotation_vector": "rvec",
        "translation_vector": "tvec"
    }

    name = aliases.get(
        name,
        name
    )

    allowed = {
        "id",
        "x",
        "y",
        "z",
        "z_depth",
        "distance",
        "omega",
        "rvec",
        "tvec",
        "corners",
        "all"
    }

    if name not in allowed:

        raise ValueError(
            "Supported measurements: id, x, y, z, "
            "z_depth, distance, omega, rvec, tvec, "
            "corners, all."
        )

    if attempts < 1:
        raise ValueError(
            "attempts must be at least 1."
        )

    ci.get_camera_info(camera_name)
    ci.load_calibration(camera_name)

    cap, frame = open_capture(camera_name)

    try:

        for attempt in range(attempts):

            if attempt > 0:

                success, frame = cap.read()

                if not success or frame is None:
                    continue

            results = detect(
                frame,
                camera_name
            )

            if marker_id is not None:

                results = [
                    item for item in results
                    if item["id"] == int(marker_id)
                ]

            if not results:
                continue

            if name == "all":

                return [
                    {
                        key: value
                        for key, value in item.items()
                        if key != "corners"
                    }
                    for item in results
                ]

            item = results[0]

            if name == "z":
                return item["distance"]

            return item[name]

        raise RuntimeError(
            f"No ArUco marker detected using {camera_name} "
            f"after {attempts} frames."
        )

    finally:

        cap.release()


# ============================================================
# LIVE ARUCO DETECTION
# ============================================================

def live(camera_name, marker_id=None):

    camera_name = camera_name.lower().strip()

    info = ci.get_camera_info(
        camera_name
    )

    ci.load_calibration(
        camera_name
    )

    cap, frame = open_capture(
        camera_name
    )

    detector_bundle = create_detector()

    print("\nArUco live detection")
    print("Camera:", info["name"])
    print("Marker size:", MARKER_SIZE_MM, "mm")
    print("Dictionary: DICT_5X5_50")

    print(
        "Marker filter:",
        marker_id if marker_id is not None
        else "None - detect all IDs"
    )

    print("Press Q in the video window to quit.\n")

    try:

        while True:

            # 1. Remove lens distortion
            corrected = dist.undistort_frame(
                frame,
                camera_name
            )

            # 2. Detect markers and calculate pose
            results = estimate_marker_poses(
                corrected,
                camera_name,
                detector_bundle,
                marker_size_mm=MARKER_SIZE_MM,
                marker_id=marker_id
            )

            # 3. Draw marker boundaries and IDs
            for item in results:

                draw_marker(
                    corrected,
                    item
                )

            # 4. Draw vertical measurements
            output_frame = draw_measurements(
                corrected,
                results
            )

            # 5. Display camera frame
            cv2.imshow(
                WINDOW_NAME,
                output_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            # 6. Read next camera frame
            success, next_frame = cap.read()

            if success and next_frame is not None:
                frame = next_frame

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print("ArUco live detection stopped.")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="ArUco detection and pose estimation"
    )

    parser.add_argument(
        "camera",
        help="Camera name or 'list'"
    )

    parser.add_argument(
        "action",
        nargs="?",
        help="Measurement name or live"
    )

    parser.add_argument(
        "--marker-id",
        type=int,
        default=None,
        help="Optional marker ID filter"
    )

    parser.add_argument(
        "--attempts",
        type=int,
        default=30
    )

    args = parser.parse_args()

    # List supported cameras
    if args.camera.lower() == "list":

        print("Supported cameras:")

        for name in ci.list_cameras():
            print(name)

        return

    camera_name = args.camera.lower().strip()

    try:

        info = ci.get_camera_info(
            camera_name
        )

        if args.action is None:

            parser.error(
                "Specify a measurement or 'live'."
            )

        action = args.action.lower().strip()

        # Camera information
        if action == "info":

            print(ci.get_camera_info(camera_name))
            return

        if action == "index":

            print(ci.get_camera_index(camera_name))
            return

        # Intrinsic camera parameters
        if action in {
            "fx", "fy", "cx", "cy",
            "matrix", "camera_matrix",
            "hfov", "vfov", "dfov",
            "fov", "distortion", "coefficients"
        }:

            if action == "fov":
                print(ci.get_fov(camera_name))

            elif action == "distortion":
                print(
                    ci.get_distortion_coefficients(
                        camera_name
                    )
                )

            elif action == "coefficients":
                print(
                    ci.get_distortion_coefficients(
                        camera_name
                    )
                )

            else:
                print(
                    ci.get_value(
                        camera_name,
                        action
                    )
                )

            return

        # Live detection
        if action == "live":

            live(
                camera_name,
                marker_id=args.marker_id
            )

            return

        # Other measurements
        result = get_value(
            camera_name,
            action,
            marker_id=args.marker_id,
            attempts=args.attempts
        )

        if action == "all":

            print("Camera:", info["name"])

            if not result:

                print("No markers detected.")
                return

            print(
                "Detected markers:",
                len(result)
            )

            for item in result:

                print(
                    "\nMarker ID:",
                    item["id"]
                )

                for key, value in item.items():

                    if key != "id":
                        print(
                            f"{key}: {value}"
                        )

        else:

            print("Camera:", info["name"])
            print("Measurement:", action)
            print("Result:", result)

    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        cv2.error
    ) as error:

        parser.exit(
            1,
            f"ERROR: {error}\n"
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()