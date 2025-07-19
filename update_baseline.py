import os
import shutil
import requests
from detect_wetness import BASELINE_IMG, DATA_DIR
from dotenv import load_dotenv

load_dotenv()
camera_ip = os.environ.get("CAMERA_IP")
camera_user = os.environ.get("CAMERA_USER")
camera_pass = os.environ.get("CAMERA_PASS")

def backup_baseline():
    if os.path.exists(BASELINE_IMG):
        backup_path = BASELINE_IMG + ".bak"
        shutil.copy2(BASELINE_IMG, backup_path)
        print(f"Previous baseline backed up to {backup_path}")

if __name__ ==  "__main__":
    if not camera_ip or not camera_user or not camera_pass:
        print("Please set CAMERA_IP, CAMERA_USER, and CAMERA_PASS in your .env file.")
        exit(1)

    try: 
        url = f"http://{camera_ip}/cgi-bin/snapshot.cgi?channel=1"
        response = requests.get(url, auth=(camera_user, camera_pass))
        response.raise_for_status()
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        backup_baseline()

        with open(BASELINE_IMG, "wb") as f:
            f.write(response.content)
        print(f"Baseline image saved to {BASELINE_IMG}")
    except requests.RequestException as e:
        print(f"Failed to download baseline image: {e}")
        exit(1)
