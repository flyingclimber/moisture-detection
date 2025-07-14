
import cv2
import numpy as np
import requests
import os
from dotenv import load_dotenv

load_dotenv()

camera_ip = os.environ.get("CAMERA_IP")
camera_user = os.environ.get("CAMERA_USER")
camera_pass = os.environ.get("CAMERA_PASS")

if not camera_ip or not camera_user or not camera_pass:
    print("Please set CAMERA_IP, CAMERA_USER, and CAMERA_PASS in your .env file.")
    exit(1)

def download_snapshot(url, auth):
    try:
        response = requests.get(url, auth=auth, stream=True, timeout=10)
        response.raise_for_status()
        with open("snapshot.jpg", "wb") as f:
            f.write(response.content)
    except Exception as e:
        print(f"Failed to download snapshot: {e}")
        exit(1)

def check_file_exists(filename):
    if not os.path.exists(filename):
        print(f"{filename} not found. Please place it in the script directory and rerun this script.")
        exit(1)
    if os.path.getsize(filename) == 0:
        print(f"{filename} is empty. Please check the file and try again.")
        exit(1)


def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"{filename} could not be loaded or is not a valid image. Please check the file and try again.")
        exit(1)
    return img


def main():
    url = f"http://{camera_ip}/cgi-bin/snapshot.cgi?channel=1"
    auth = (camera_user, camera_pass)

    download_snapshot(url, auth)

    check_file_exists("baseline.jpg")
    check_file_exists("snapshot.jpg")

    baseline = load_image("baseline.jpg")
    current = load_image("snapshot.jpg")

    if current.shape != baseline.shape:
        current = cv2.resize(current, (baseline.shape[1], baseline.shape[0]))

    diff = cv2.absdiff(current, baseline)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    changed_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.size
    percent_changed = (changed_pixels / total_pixels) * 100

    print(f"Changed pixels: {percent_changed:.2f}%")
    cv2.imwrite("diff.jpg", thresh)

    if percent_changed > 2.5:
        print("⚠️ Wetness detected!")

if __name__ == "__main__":
    main()
