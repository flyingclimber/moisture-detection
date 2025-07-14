import cv2
import numpy as np
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Camera settings
camera_ip = os.environ.get("CAMERA_IP")
camera_user = os.environ.get("CAMERA_USER")
camera_pass = os.environ.get("CAMERA_PASS")

if not camera_ip or not camera_user or not camera_pass:
    raise ValueError("CAMERA_IP, CAMERA_USER and CAMERA_PASS environment variables must be set.")
url = f"http://{camera_ip}/cgi-bin/snapshot.cgi?channel=1"
auth = (camera_user, camera_pass)

# Download current snapshot
response = requests.get(url, auth=auth, stream=True)
with open("snapshot.jpg", "wb") as f:
    f.write(response.content)

# Check for baseline image
if not os.path.exists("baseline.jpg"):
    print("baseline.jpg not found. Please place a baseline image in the script directory and rerun this script.")
    exit(1)

# Check that snapshot.jpg is not zero bytes
if not os.path.exists("snapshot.jpg") or os.path.getsize("snapshot.jpg") == 0:
    print("snapshot.jpg is missing or empty. Please check the camera connection and try again.")
    exit(1)

# Load images
baseline = cv2.imread("baseline.jpg", cv2.IMREAD_GRAYSCALE)
current = cv2.imread("snapshot.jpg", cv2.IMREAD_GRAYSCALE)

# Validate snapshot image
if current is None:
    print("snapshot.jpg could not be loaded or is not a valid image. Please check the camera connection and try again.")
    exit(1)

# Resize if needed
if current.shape != baseline.shape:
    current = cv2.resize(current, (baseline.shape[1], baseline.shape[0]))

# Compute diff
diff = cv2.absdiff(current, baseline)
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Calculate changed %
changed_pixels = np.count_nonzero(thresh)
total_pixels = thresh.size
percent_changed = (changed_pixels / total_pixels) * 100

print(f"Changed pixels: {percent_changed:.2f}%")

# Save diff image
cv2.imwrite("diff.jpg", thresh)

if percent_changed > 2.5:
    print("⚠️ Wetness detected!")
