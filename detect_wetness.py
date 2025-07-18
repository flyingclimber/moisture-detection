import cv2
import numpy as np
import requests
import os
import argparse
import logging
import datetime
import json
from dotenv import load_dotenv
from dateutil.parser import parse as parse_date

DATA_DIR: str = "data"
LOG_FILE: str = os.path.join(DATA_DIR, "detect_wetness.log")
BASELINE_IMG: str = os.path.join(DATA_DIR, "baseline.jpg")
SNAPSHOT_IMG: str = os.path.join(DATA_DIR, "snapshot.jpg")
DIFF_IMG: str = os.path.join(DATA_DIR, "diff.jpg")
WETNESS_THRESHOLD: float = 2.5
THRESHOLD_VALUE: int = 30
WEATHER_POINTS_URL = "https://api.weather.gov/points"
RAIN_THRESHOLD: int = 50 
STATE_FILE = os.path.join(DATA_DIR, "state.json")
N_HOURS = int = 6 

load_dotenv()
camera_ip = os.environ.get("CAMERA_IP")
camera_user = os.environ.get("CAMERA_USER")
camera_pass = os.environ.get("CAMERA_PASS")
slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
loc = os.environ.get("LOC_GPS", "37.773,-122.431")

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

def is_rain_forecasted(state) -> bool:
    """
    Check if rain is forecasted in the current hourly period.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    valid_until = state.get('forecast_valid_until')
    rain_forecasted = state.get('rain_forecasted')
    if valid_until:
        try:
            valid_until_dt = parse_date(valid_until)
            if now < valid_until_dt:
                logger.info(f"Using cached rain forecast (valid until {valid_until})")
                return rain_forecasted
        except Exception as e:
            logger.warning(f"Could not parse cached forecast_valid_until: {e}")

    # Fetch new forecast
    try:
        forecast_url = f"{WEATHER_POINTS_URL}/{loc}"
        response = requests.get(forecast_url, timeout=10)
        response.raise_for_status()
        forecast_hourly_url = response.json().get("properties", {}).get("forecastHourly")
        if not forecast_hourly_url:
            logger.error("No hourly forecast URL found in weather data.")
            return False

        forecast_response = requests.get(forecast_hourly_url, timeout=10)
        forecast_response.raise_for_status()
        periods = forecast_response.json().get("properties", {}).get("periods", [])

        rain_found = False
        latest_end = now
        for period in periods:
            start = parse_date(period.get('startTime'))
            end = parse_date(period.get('endTime'))
            if start > now + datetime.timedelta(hours=N_HOURS):
                break
            if end > latest_end:
                latest_end = end
            precip = period.get('probabilityOfPrecipitation', {}).get('value', 0)
            if precip >= RAIN_THRESHOLD:
                rain_found = True

        state['forecast_valid_until'] = latest_end.isoformat()
        state['rain_forecasted'] = rain_found
        state['last_forecast_check'] = now.isoformat()
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Fetched new rain forecast (valid until {latest_end.isoformat()})")
        return rain_found
    except Exception as e:
        logger.error(f"Failed to fetch weather data: {e}")
    return False


def check_lights_on(image: np.ndarray, mean_threshold: float = 100.0) -> bool:
    """
    Return True if the mean pixel value is above the threshold (i.e., image is very bright).
    """
    mean_brightness = np.mean(image)
    return mean_brightness > mean_threshold

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

    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
        else:
            state = {}
    except Exception:
        state = {}

    rain_forecasted = is_rain_forecasted(state)
    state['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if not rain_forecasted:
        logger.info("No rain is forecasted, skipping wetness detection.")
        state['skipped'] = 'no_rain'
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        return

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

    baseline = load_image(baseline_path)[600:, 0:]
    current = load_image(snapshot_path)[600:, 0:]

    lights_on = check_lights_on(current)
    state['lights_on'] = lights_on

    # Early exit if lights are on in the snapshot
    if lights_on:
        logger.info("Lights are on, skipping wetness detection.")
        state['skipped'] = 'lights_on'
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        return

    if current.shape != baseline.shape:
        current = cv2.resize(current, (baseline.shape[1], baseline.shape[0]))

    diff = cv2.absdiff(current, baseline)
    _, thresh = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    changed_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.size
    percent_changed = (changed_pixels / total_pixels) * 100
    state['percent_changed'] = percent_changed

    if percent_changed > WETNESS_THRESHOLD:
        alert_msg = f"⚠️ Wetness detected! {percent_changed:.2f}% of pixels changed."
        logger.warning(alert_msg)
        notify_slack(alert_msg)
        state['wetness_detected'] = True
    else:
        logger.info(f"Changed pixels: {percent_changed:.2f}%")
        state['wetness_detected'] = False
    cv2.imwrite(diff_img_path, thresh)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

if __name__ == "__main__":
    main()
