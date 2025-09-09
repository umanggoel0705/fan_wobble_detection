import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.signal import find_peaks
from ultralytics import YOLO
import time

analysis_data = {}
ROI = ()
FPS = None
inference_cache = []

# Remove consecutive duplicates
def remove_duplicates(dist_list):
    dist = [key for key, _ in groupby(dist_list)]
    return dist

def plot(x_plot, y_plot, dist_plot, format:str):
    # Remove consecutive duplicates
    X_plot = remove_duplicates(x_plot)
    Y_plot = remove_duplicates(y_plot)
    Dist_plot = remove_duplicates(dist_plot)

# Create a 1x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2 rows, 2 columns

    axs[0, 0].set_title("X & Y distances from intial center")
    axs[0, 0].plot(x_plot, label='X')
    axs[0, 0].plot(y_plot, label='Y')
    axs[0, 0].set_xlabel(f"No. of {format}")
    axs[0, 0].legend()
    # plt.show()

    axs[0, 1].set_title("Cartesian distance from initial center")
    axs[0, 1].plot(dist_plot, label='Distance')
    axs[0, 1].set_xlabel(f"No. of {format}")
    axs[0, 1].legend()

    axs[1, 0].set_title("X & Y distances from intial center w/o duplicates")
    axs[1, 0].plot(X_plot, label='X')
    axs[1, 0].plot(Y_plot, label='Y')
    axs[1, 0].set_xlabel(f"No. of {format}")
    axs[1, 0].legend()
    # plt.show()

    axs[1, 1].set_title("Cartesian distance from initial center w/o duplicates")
    axs[1, 1].plot(Dist_plot, label='Distance')
    axs[1, 1].set_xlabel(f"No. of {format}")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"static/distance_plots_wrt_frames.png")
    plt.close()

# Convert from per frame to per sec
def convert_to_sec(x_dist, y_dist, dist, fps):
    # Define block size for which we want the mean (fps values per block) to convert into /sec format
    block_size = int(fps)

    # Initialize an empty list to store the means
    x_means = []
    y_means = []
    dist_means = []

    # Iterate over the data in steps of block_size
    for i in range(0, len(x_dist), block_size):
        # Get the next block of block_size values (or fewer if near the end)
        x_block = x_dist[i:i + block_size]
        y_block = y_dist[i:i + block_size]
        dist_block = dist[i:i + block_size]
        
        # Calculate the mean of the current block
        x_block_mean = np.mean(x_block)
        y_block_mean = np.mean(y_block)
        dist_block_mean = np.mean(dist_block)
        
        # Append the mean to the list of means
        x_means.append(x_block_mean)
        y_means.append(y_block_mean)
        dist_means.append(dist_block_mean)

    plot(x_means, y_means, dist_means, "seconds")

def cartesian_dist_analysis(dist):
    Dist_plot = remove_duplicates(dist)

    # replace this with distance list or array
    distance = np.array(Dist_plot)

    # Step 1: Find peaks
    peaks, _ = find_peaks(distance)

    # Step 2: Find troughs (invert the signal)
    troughs, _ = find_peaks(-distance)

    # Step 3: Count of peaks and troughs
    peak_count = len(peaks)
    trough_count = len(troughs)

    # Step 4: Pair up peaks and troughs to compute amplitudes
    amplitudes = []

    # Loop through peaks and find the nearest preceding trough
    for peak in peaks:
        preceding_troughs = troughs[troughs < peak]
        if len(preceding_troughs) > 0:
            last_trough = preceding_troughs[-1]
            amp = distance[peak] - distance[last_trough]
            amplitudes.append(int(amp))

    amplitudes = np.array(amplitudes)

    # print(f"Number of peaks: {peak_count}")
    # print(f"Number of troughs: {trough_count}")
    # print(f"Calculated amplitudes (peak - preceding trough): {amplitudes}")
    # print(f"Mean of amplitudes: {amplitudes.mean()}")
    # print(f"Median of amplitudes: {np.median(amplitudes)}")

    global analysis_data
    analysis_data = {
        'peak_count': peak_count,
        'trough_count': trough_count,
        'amplitude_mean': round(amplitudes.mean(), 2),
        'amplitude_median': np.median(amplitudes)
    }

    # Plot to visualize
    plt.title("Cartesian Distance w/o duplicates")
    plt.plot(distance, label='Distance')
    plt.xlabel("No. of Frames")
    plt.legend()
    plt.savefig(f"static/cartesian_analysis.png")
    plt.close()

def tracker(video_path, roi_bbox=None):
    # Add roi_bbox as parameter, just to maintain the structure of app.py, it is not being used here

    global inference_cache
    if not inference_cache:
        raise RuntimeError("main() must be called before tracker().")

    x_dist = []
    y_dist = []
    dist = []

    # Get initial center from first frame
    first_frame, first_results = inference_cache[0]
    x_ini, y_ini = first_results[0].boxes.xywh[0][:2].cpu().numpy()
    ini_center = (int(x_ini), int(y_ini))

    for frame, results in inference_cache:
        frame = frame.copy()

        # Draw bounding boxes
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Current center
        x_curr, y_curr = results[0].boxes.xywh[0][:2].cpu().numpy()
        center = (int(x_curr), int(y_curr))

        # Compute distance from initial center
        x_val = center[0] - ini_center[0]
        y_val = center[1] - ini_center[1]

        x_dist.append(x_val)
        y_dist.append(y_val)
        dist.append(math.sqrt(x_val ** 2 + y_val ** 2))

        # Draw centers
        cv2.circle(frame, center, 5, (0, 0, 255), -1)        # red current center
        cv2.circle(frame, ini_center, 5, (255, 255, 255), -1)  # white initial center

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        # Yield in MJPEG format
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(1 / FPS) # To maintain original speed

    plot(x_dist, y_dist, dist, "frames")
    # convert_to_sec(x_dist, y_dist, dist, fps)
    cartesian_dist_analysis(dist)

def tracker_roi(video_path, roi_bbox=None):
    global inference_cache
    if not inference_cache:
        raise RuntimeError("main() must be called before tracker_roi().")

    # Get initial center from first frame
    first_frame, first_results = inference_cache[0]
    x_ini, y_ini = first_results[0].boxes.xywh[0][:2].cpu().numpy()
    ini_center = (int(x_ini), int(y_ini))

    for frame, results in inference_cache:
        frame = frame.copy()

        # Draw bounding boxes
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Default message frame (in case tracking fails)
        cropped_frame = np.zeros_like(frame)
        cv2.putText(cropped_frame, "Tracking failure detected", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if results[0].boxes.xywh.shape[0] > 0:
            x, y, w, h = [int(v) for v in results[0].boxes.xywh[0].cpu().numpy()]
            center = (x, y)

            # Draw centers
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.circle(frame, ini_center, 5, (255, 255, 255), -1)

            # Crop around object
            x1 = max(x - w // 2 - 10, 0)
            x2 = min(x + w // 2 + 10, frame.shape[1])
            y1 = max(y - h // 2 - 10, 0)
            y2 = min(y + h // 2 + 10, frame.shape[0])
            cropped_frame = frame[y1:y2, x1:x2]

        # Encode frame
        _, buffer = cv2.imencode('.jpg', cropped_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(1 / FPS) # To maintain original speed


def main(video_path):
    
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Original FPS:", fps)
    global FPS
    FPS = int(fps)

    global inference_cache
    inference_cache = []  # clear cache on new run

    # Load the trained model
    model = YOLO("../runs/detect/train/weights/best.pt")

    # Load video
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        _, frame = cap.read()
        if not _:
            break

        # Run inference
        results = model(frame, verbose=False)
        inference_cache.append((frame.copy(), results))  # Cache results

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
