import cv2
import numpy as np
import requests
import os
import argparse
import logging
from dotenv import load_dotenv

DATA_DIR: str = "data"
LOG_FILE: str = os.path.join(DATA_DIR, "detect_wetness.log")
BASELINE_IMG: str = os.path.join(DATA_DIR, "baseline.jpg")
SNAPSHOT_IMG: str = os.path.join(DATA_DIR, "snapshot.jpg")
DIFF_IMG: str = os.path.join(DATA_DIR, "diff.jpg")
WETNESS_THRESHOLD: float = 2.5
THRESHOLD_VALUE: int = 30

load_dotenv()
camera_ip = os.environ.get("CAMERA_IP")
camera_user = os.environ.get("CAMERA_USER")
camera_pass = os.environ.get("CAMERA_PASS")
slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

def setup_logger(log_file: str) -> logging.Logger:
    """
    Set up and return a logger that logs to both file and console.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("detect_wetness")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger(LOG_FILE)

if not camera_ip or not camera_user or not camera_pass:
    logger.error("Please set CAMERA_IP, CAMERA_USER, and CAMERA_PASS in your .env file.")
    exit(1)

def download_snapshot(url: str, auth: tuple[str, str]) -> None:
    """
    Download a snapshot image from the camera and save it to SNAPSHOT_IMG.
    Exits on failure.
    """
    try:
        response = requests.get(url, auth=auth, stream=True, timeout=10)
        response.raise_for_status()
        with open(SNAPSHOT_IMG, "wb") as f:
            f.write(response.content)
    except Exception as e:
        logger.error(f"Failed to download snapshot: {e}")
        exit(1)

def check_lights_on(image: np.ndarray, brightness_threshold: float = 200.0, pixel_fraction: float = 0.5) -> bool:
    """
    Detect if the lights are on by checking if a large fraction of pixels are very bright.
    Args:
        image: Grayscale image as numpy array.
        brightness_threshold: Pixel value above which a pixel is considered 'bright'.
        pixel_fraction: Fraction of pixels that must be bright to consider lights on.
    Returns:
        True if lights are likely on, False otherwise.
    """
    bright_pixels = np.count_nonzero(image > brightness_threshold)
    total_pixels = image.size
    if total_pixels == 0:
        return False
    return (bright_pixels / total_pixels) >= pixel_fraction

def notify_slack(message: str) -> None:
    """
    Send a notification to Slack using the global slack_webhook_url.
    Args:
        message: The message to send.
    """
    if not slack_webhook_url:
        logger.warning("Slack webhook URL not set. Skipping Slack notification.")
        return
    payload = {"text": message}
    try:
        response = requests.post(slack_webhook_url, json=payload)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")

def check_file_exists(filename: str) -> None:
    """
    Check if a file exists and is not empty. Exits on failure.
    """
    if not os.path.exists(filename):
        logger.error(f"{filename} not found. Please place it in the script directory and rerun this script.")
        exit(1)
    if os.path.getsize(filename) == 0:
        logger.error(f"{filename} is empty. Please check the file and try again.")
        exit(1)

def load_image(filename: str) -> np.ndarray:
    """
    Load an image in grayscale. Exits if the image is invalid or cannot be loaded.
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error(f"{filename} could not be loaded or is not a valid image. Please check the file and try again.")
        exit(1)
    return img

def main() -> None:
    """
    Main workflow: download snapshot or use custom, check files, compare images, and report wetness.
    """
    parser = argparse.ArgumentParser(description="Detect wetness by comparing a camera snapshot to a baseline image.")
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Path to a custom snapshot image. If not provided, a snapshot will be downloaded from the camera."
    )
    args = parser.parse_args()

    # Resolve snapshot path
    if args.snapshot:
        snapshot_path = args.snapshot
    else:
        url = f"http://{camera_ip}/cgi-bin/snapshot.cgi?channel=1"
        auth = (camera_user, camera_pass)
        download_snapshot(url, auth)
        snapshot_path = SNAPSHOT_IMG

    # Always use DATA_DIR for baseline and diff images
    baseline_path = BASELINE_IMG
    diff_img_path = DIFF_IMG

    check_file_exists(baseline_path)
    check_file_exists(snapshot_path)

    baseline = load_image(baseline_path)
    current = load_image(snapshot_path)

    # Early exit if lights are on in the snapshot
    if check_lights_on(current):
        logger.info("Lights are on, skipping wetness detection.")
        return

    if current.shape != baseline.shape:
        current = cv2.resize(current, (baseline.shape[1], baseline.shape[0]))

    diff = cv2.absdiff(current, baseline)
    _, thresh = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    changed_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.size
    percent_changed = (changed_pixels / total_pixels) * 100

    if percent_changed > WETNESS_THRESHOLD:
        alert_msg = f"⚠️ Wetness detected! {percent_changed:.2f}% of pixels changed."
        logger.warning(alert_msg)
        notify_slack(alert_msg)
    else:
        logger.info(f"Changed pixels: {percent_changed:.2f}%")
    cv2.imwrite(diff_img_path, thresh)

if __name__ == "__main__":
    main()
