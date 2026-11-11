"""
distortion.py

Reusable lens-distortion library. Camera selection is by camera name and
physical USB VID/PID identity. The current OpenCV index is discovered at runtime.
"""

import cv2
import numpy as np
import argparse

import camera_intrinsics as ci


CAMERAS = ci.list_cameras()
_MAP_CACHE = {}
COEFFICIENT_NAMES = ["k1", "k2", "p1", "p2", "k3"]


def get_coefficients(camera_name):
    """Return all distortion coefficients for one camera."""
    return ci.get_distortion_coefficients(camera_name).flatten()


def get_value(camera_name, value_name):
    """Return one coefficient or all coefficients."""
    name = value_name.lower().strip()
    coefficients = get_coefficients(camera_name)

    if name in ("all", "coefficients", "distortion_coefficients"):
        return {
            key: float(value)
            for key, value in zip(COEFFICIENT_NAMES, coefficients)
        }

    if name not in COEFFICIENT_NAMES:
        raise ValueError(
            "Supported values: k1, k2, p1, p2, k3, "
            "coefficients, distortion_coefficients, all."
        )

    return float(coefficients[COEFFICIENT_NAMES.index(name)])


def get_info(camera_name):
    """Return camera identity, image settings and distortion coefficients."""
    info = ci.get_camera_info(camera_name)
    coefficients = get_coefficients(camera_name)

    return {
        "camera": info["name"],
        "camera_key": camera_name.lower().strip(),
        "usb_vid": info["vid"],
        "usb_pid": info["pid"],
        "width": info["width"],
        "height": info["height"],
        "coefficients": {
            key: float(value)
            for key, value in zip(COEFFICIENT_NAMES, coefficients)
        },
    }


def create_maps(camera_name, width=640, height=480, alpha=0.0):
    """Create undistortion maps for the selected camera."""
    name = camera_name.lower().strip()
    ci.get_camera_info(name)

    width = int(width)
    height = int(height)
    alpha = float(alpha)

    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")

    key = (name, width, height, alpha)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]

    camera_matrix, coefficients = ci.load_calibration(name)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, coefficients, (width, height), alpha, (width, height)
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, coefficients, None, new_camera_matrix,
        (width, height), cv2.CV_32FC1
    )

    result = (map1, map2, new_camera_matrix, roi)
    _MAP_CACHE[key] = result
    return result


def undistort_frame(frame, camera_name, alpha=0.0):
    """Correct one image frame using its camera calibration."""
    if frame is None:
        raise ValueError("Input frame is None.")

    height, width = frame.shape[:2]
    map1, map2, _, _ = create_maps(camera_name, width, height, alpha)
    return cv2.remap(
        frame, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def print_all(camera_name):
    """Print camera identity and all five distortion coefficients."""
    info = ci.get_camera_info(camera_name)
    values = get_value(camera_name, "all")

    print("\n" + "=" * 50)
    print("DISTORTION INFORMATION")
    print("=" * 50)
    print("Camera:", info["name"])
    print("Camera key:", camera_name.lower().strip())
    print("USB VID:PID:", f"{info['vid']}:{info['pid']}")
    print("Resolution:", f"{info['width']} x {info['height']}")
    print("\nDistortion coefficients:")
    for name in COEFFICIENT_NAMES:
        print(f"{name}: {values[name]:.10f}")
    print("=" * 50)


def live(camera_name):
    """Display original and corrected images side by side."""
    info = ci.get_camera_info(camera_name)
    ci.load_calibration(camera_name)
    cap = ci.open_camera(camera_name)

    print("Live distortion-correction test")
    print("Camera:", info["name"])
    print("USB VID:PID:", f"{info['vid']}:{info['pid']}")
    print("Press Q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Could not read a camera frame.")
                break

            corrected = undistort_frame(frame, camera_name)
            original_view = frame.copy()
            corrected_view = corrected.copy()

            cv2.putText(original_view, "ORIGINAL", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(corrected_view, "UNDISTORTED", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            comparison = np.hstack((original_view, corrected_view))
            cv2.imshow(f"Distortion - {camera_name}", comparison)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Distortion test stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Reusable camera distortion library with automatic camera discovery"
    )
    parser.add_argument("camera", nargs="?", help="Camera name or 'list'")
    parser.add_argument(
        "action", nargs="?",
        help="k1, k2, p1, p2, k3, coefficients, all, info, or live"
    )
    args = parser.parse_args()

    if args.camera is None or args.camera.lower() == "list":
        print("Supported cameras:")
        for name in CAMERAS:
            info = ci.get_camera_info(name)
            print(f"  {name} ({info['vid']}:{info['pid']})")
        return

    camera_name = args.camera.lower().strip()
    if camera_name not in CAMERAS:
        parser.error(
            f"Unknown camera '{args.camera}'. Available: {', '.join(CAMERAS)}"
        )
    if args.action is None:
        parser.error("Specify a coefficient, all, info, or live.")

    action = args.action.lower().strip()
    try:
        if action == "live":
            live(camera_name)
        elif action == "info":
            print(get_info(camera_name))
            print("Currently discovered device:", ci.discover_camera(camera_name))
        elif action == "all":
            print_all(camera_name)
        else:
            result = get_value(camera_name, action)
            print("Camera:", ci.get_camera_info(camera_name)["name"])
            print("Value:", action)
            print("Result:", result)
    except (ValueError, RuntimeError, FileNotFoundError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
