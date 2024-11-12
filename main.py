import io
import numpy as np
import datetime
import os
import gc
from collections import deque
from flask import Flask, Response
import cv2
from threading import Thread, Lock, Event
import psutil

app = Flask(__name__)

# Global variables for recording
recording = True
video_writer = None
frame_queue = []
shutdown_event = Event()
frame_buffer = deque(maxlen=450)  # 30 seconds * 15 fps = 450 frames
buffer_lock = Lock()

# Initialize USB camera with lower resolution
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 15)

# Initialize VideoWriter
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 15.0  # Explicitly set FPS for consistency
fourcc = cv2.VideoWriter_fourcc(*'XVID')
filename = f"camera_recording_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.avi"
video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

# Ensure identification directory exists
os.makedirs("identification", exist_ok=True)


def check_memory():
    memory = psutil.Process().memory_percent()
    if memory > 70:
        with buffer_lock:
            frame_buffer.clear()
            gc.collect()
        return False
    return True


def compress_frame(frame):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    _, compressed = cv2.imencode('.jpg', frame, encode_param)
    return compressed


def decompress_frame(compressed):
    return cv2.imdecode(compressed, cv2.IMREAD_COLOR)


def capture_frames():
    global recording, video_writer, frame_queue, camera, frame_buffer
    frame_count = 0
    last_capture_time = datetime.datetime.now()
    frame_interval = 1.0 / fps  # Time between frames in seconds

    try:
        while not shutdown_event.is_set():
            if frame_count % 30 == 0:
                if not check_memory():
                    print("Memory usage high - cleared buffer")

            # Control capture timing
            current_time = datetime.datetime.now()
            elapsed = (current_time - last_capture_time).total_seconds()

            if elapsed < frame_interval:
                continue

            last_capture_time = current_time

            ret, frame = camera.read()
            if not ret:
                continue

            frame_count += 1

            # Store frame in buffer with timestamp
            with buffer_lock:
                frame_buffer.append({
                    'frame': compress_frame(frame),
                    'timestamp': current_time
                })

            if len(frame_queue) < 5:
                frame_queue.append(frame)

            if recording and video_writer:
                video_writer.write(frame)

    except Exception as e:
        print(f"Error in capture_frames: {e}")
    finally:
        cleanup()


def cleanup():
    global camera, video_writer
    if camera is not None:
        camera.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    gc.collect()


def generate_frames():
    while not shutdown_event.is_set():
        if frame_queue:
            frame = frame_queue.pop(0)
            try:
                ret, buffer = cv2.imencode('.jpg', frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if not ret:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error in generate_frames: {e}")
                continue


@app.route('/save_last_30_seconds')
def save_last_30_seconds():
    """Save the last 30 seconds of footage to the identification folder"""
    global frame_buffer

    with buffer_lock:
        if len(frame_buffer) == 0:
            return "No frames available to save"

        try:
            clip_filename = os.path.join(
                "identification",
                f"clip_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.avi"
            )

            # Create clip writer with same parameters as main recording
            clip_writer = cv2.VideoWriter(
                clip_filename,
                fourcc,
                fps,  # Use same FPS as main recording
                (width, height)
            )

            # Calculate actual frame intervals for proper timing
            buffer_frames = list(frame_buffer)
            if len(buffer_frames) >= 2:
                actual_fps = len(buffer_frames) / (
                    (buffer_frames[-1]['timestamp'] - buffer_frames[0]['timestamp']).total_seconds()
                )
                if actual_fps > 0:
                    # Adjust number of frames to write based on actual capture rate
                    frames_to_write = int(30 * actual_fps)  # 30 seconds worth of frames
                    frames_to_write = min(frames_to_write, len(buffer_frames))

                    # Write frames maintaining original timing
                    for frame_data in buffer_frames[-frames_to_write:]:
                        decompressed_frame = decompress_frame(frame_data['frame'])
                        clip_writer.write(decompressed_frame)

            clip_writer.release()
            return f"Saved last 30 seconds to: {clip_filename}"

        except Exception as e:
            return f"Error saving clip: {e}"


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_recording')
def start_recording():
    global recording, video_writer
    if not recording:
        try:
            new_filename = f"camera_recording_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.avi"
            video_writer = cv2.VideoWriter(new_filename, fourcc, fps, (width, height))
            recording = True
            return f"Recording started! Saving to: {new_filename}"
        except Exception as e:
            return f"Error starting recording: {e}"
    return "Already recording!"


@app.route('/stop_recording')
def stop_recording():
    global recording, video_writer
    if recording:
        recording = False
        if video_writer:
            video_writer.release()
            video_writer = None
        return "Recording stopped!"
    return "Not recording!"


@app.route('/shutdown')
def shutdown():
    shutdown_event.set()
    cleanup()
    return "Server shutting down..."


if __name__ == '__main__':
    try:
        print(f"Started initial recording to file: {filename}")
        print("Access http://localhost:5000/video_feed to see the live stream")
        print("Use http://localhost:5000/stop_recording to stop recording")
        print("Use http://localhost:5000/start_recording to start a new recording")
        print("Use http://localhost:5000/save_last_30_seconds to save the last 30 seconds")
        print("Use http://localhost:5000/shutdown for graceful shutdown")

        capture_thread = Thread(target=capture_frames)
        capture_thread.daemon = True
        capture_thread.start()

        app.run(host='0.0.0.0', port=5000)

    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        shutdown_event.set()
        cleanup()
