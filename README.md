# Moisture Detection Script

This script detects wetness by comparing a camera snapshot to a baseline image using OpenCV.

## Features
- Downloads a snapshot from an IP camera or uses a custom snapshot image.
- Compares the snapshot to a baseline image and calculates the percentage of changed pixels.
- Alerts if wetness is detected based on a configurable threshold.

## Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Requests
- python-dotenv

Install dependencies:
```
pip install opencv-python numpy requests python-dotenv
```

You might need the following apt packages if you install from pinwheel
```
apt-get install -y libatlas3-base libavcodec59 libavformat59 libgtk-3-0 libopenblas0 libopenjp2-7 libswscale6
```

## Usage

1. Place a baseline image named `baseline.jpg` in the script directory.
2. Create a `.env` file with your camera credentials:
    ```
    CAMERA_IP=192.168.1.100
    CAMERA_USER=admin
    CAMERA_PASS=yourpassword
    ```
3. Run the script:
    ```
    python detect_wetness.py
    ```
   Or use a custom snapshot image:
    ```
    python detect_wetness.py --snapshot path/to/image.jpg
    ```

## Configuration
- Change the constants at the top of `detect_wetness.py` to adjust filenames or thresholds.
- `WETNESS_THRESHOLD` sets the percentage of changed pixels to trigger a wetness alert.
- `THRESHOLD_VALUE` sets the pixel difference threshold for change detection.

## Output
- Prints the percentage of changed pixels.
- Saves a binary difference image as `diff.jpg`.
- Prints a warning if wetness is detected.

## License
MIT
