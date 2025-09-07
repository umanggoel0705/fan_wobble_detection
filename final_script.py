import math
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for image generation
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.signal import find_peaks
import os

analysis_data = {}
ROI = ()
FPS = None

# Remove consecutive duplicates
def remove_duplicates(dist_list):
    dist = [key for key, _ in groupby(dist_list)]
    return dist

def plot(x_plot, y_plot, dist_plot, format:str):
    # Remove consecutive duplicates
    X_plot = remove_duplicates(x_plot)
    Y_plot = remove_duplicates(y_plot)
    Dist_plot = remove_duplicates(dist_plot)

    os.makedirs("static", exist_ok=True)
    
    # Create a 2x2 grid of plots
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

def tracker(video_path: str, bbox):
    x_dist = []
    y_dist = []
    dist = []

    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()

    # Get the original frame rate (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Original FPS:", fps)

    # Manually select the initial bounding box around the fan(center part) in the first frame
    # bbox = cv2.selectROI("Select Fan", frame, fromCenter=True)
    x, y, w, h = [int(v) for v in bbox]

    print("ROI in tracker:", bbox)

    ini_center = (x + w // 2, y + h // 2) # Initial Center point
    cv2.destroyWindow("Select Fan")

    # Initialize a tracker (e.g., KCF or MOSSE)
    tracker = cv2.TrackerKCF_create() 
    tracker.init(frame, bbox)


    while cap.isOpened():
        _, frame = cap.read()
        if not _:
            break

        # Update the tracker and get the new bounding box
        success, bbox = tracker.update(frame)
        
        if success:
            # Draw the bounding box around the fan
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculate the center of the bounding box
            center = (x + w // 2, y + h // 2)

            # Distance from initial center point
            x_val = (center[0] - ini_center[0]) 
            y_val = (center[1] - ini_center[1])

            # x_val = center[0]
            # y_val = center[1]

            x_dist.append(x_val) 
            y_dist.append(y_val)
            dist.append(math.sqrt(x_val**2 + y_val**2))

            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.circle(frame, ini_center, 5, (255, 255, 255), -1)
        
        else:
            # If the tracker fails to track the object, display a failure message
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # # Display the frame with the tracking information
        # cv2.imshow("Tracking Fan", frame)
        
        # # The delay is calculated as (1000 / fps), so the frame rate is preserved
        # if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
        #     break

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        # Yield in MJPEG format
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

    plot(x_dist, y_dist, dist, "frames")
    cartesian_dist_analysis(dist)

def tracker_roi(video_path: str, roi_bbox):
    print("ROI in tracker_roi:", roi_bbox)

    x, y, w, h = [int(v) for v in roi_bbox]
    ini_center = (x + w // 2, y + h // 2) # Initial Center point
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()

    # Initialize tracker with given roi_bbox, no selection here
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, roi_bbox)

    while cap.isOpened():
        _, frame = cap.read()
        if not _:
            break

        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            # Crop only the ROI area from frame
            cropped_frame = frame[y-10:y+h+10, x-10:x+w+10]
            
            # Calculate the center of the bounding box
            center = (10 + (w // 2), 10 + (h // 2))
            
            # Draw bbox on cropped frame 
            cv2.rectangle(cropped_frame, (10, 10), (w, h), (0, 255, 0), 2)
            cv2.circle(cropped_frame, center, 5, (0, 0, 255), -1)
            cv2.circle(frame, ini_center, 5, (255, 255, 255), -1)
        else:
            # If the tracker fails to track the object, display a failure message
            cv2.putText(cropped_frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', cropped_frame)
        frame_bytes = buffer.tobytes()
        # Yield in MJPEG format
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def main(video_path):
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS)

    global FPS
    FPS = int(fps)

    bbox = cv2.selectROI("Select Fan", frame, fromCenter=True)

    global ROI
    ROI = tuple(bbox)
    print("ROI in main:", ROI)

    tracker(video_path, ROI)

    # global analysis_data
    # return analysis_data
    