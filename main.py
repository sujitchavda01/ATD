import cv2
import pandas as pd
from deepface import DeepFace
import os
import time
from datetime import datetime
import logging

# --- Configuration ---
DB_PATH = "my_db"
LOGS_PATH = "logs"
LOG_FILE = os.path.join(LOGS_PATH, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv")
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface" # Very accurate, but slower. Use 'opencv' or 'ssd' for speed if needed.
FRAME_SKIP = 5 # Process every Nth frame
COOL_OFF_MINUTES = 30

# --- Global State ---
attendance_log = {} # {name: last_log_time_timestamp}

# --- Setup Logging ---
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)

# Check if DB exists and is populated
if not os.path.exists(DB_PATH) or not [d for d in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, d))]:
    print(f"WARNING: Database folder '{DB_PATH}' appears empty or missing subfolders.")
    print("Please create subfolders named after individuals (e.g., my_db/John_Doe) and add images.")

def log_attendance(name):
    """
    Logs the person's attendance to CSV if the cool-off period has passed.
    """
    now = time.time()
    last_seen = attendance_log.get(name)
    
    if last_seen and (now - last_seen) < (COOL_OFF_MINUTES * 60):
        return False # Cooled off, didn't log

    # Log to file
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_record = pd.DataFrame({'Name': [name], 'Time': [timestamp_str]})
    
    if not os.path.isfile(LOG_FILE):
        new_record.to_csv(LOG_FILE, mode='w', header=True, index=False)
    else:
        new_record.to_csv(LOG_FILE, mode='a', header=False, index=False)
    
    attendance_log[name] = now
    print(f"Logged: {name} at {timestamp_str}")
    return True

def recognize_frame(frame):
    """
    Performs face recognition on a single frame.
    Returns a list of identified names and their bounding boxes.
    Structure: [{'name': 'Alice', 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}}]
    """
    results = []
    
    try:
        # DeepFace.find returns a list of DataFrames (one per face detected)
        # We pass silent=True to suppress exceptions if no face is found
        dfs = DeepFace.find(
            img_path=frame,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            silent=True
        )
        
        for df in dfs:
            if not df.empty:
                # Get the most similar match (first row usually, as it's sorted by distance)
                match = df.iloc[0]
                source_path = match['identity'] # Full path: my_db/Name/image.jpg
                
                # Extract name from folder name
                # distinct path separators handling
                name = os.path.basename(os.path.dirname(source_path))
                
                # Extract coordinates
                region = match['source_x'], match['source_y'], match['source_w'], match['source_h']
                
                results.append({
                    'name': name,
                    'region': region
                })
                
    except Exception as e:
        # DeepFace might raise errors if something goes wrong internally
        # print(f"Recognition error: {e}") 
        pass
        
    return results

def draw_overlay(frame, recognition_results, fps):
    """
    Draws bounding boxes, names, and UI elements on the frame.
    """
    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    for res in recognition_results:
        name = res['name']
        x, y, w, h = res['region']
        
        # Color based on cool-off status (Visual feedback)
        # If recently logged, maybe Green, else Blue? 
        # For now, consistent Green box
        color = (0, 255, 0)
        
        # Rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Name tag background
        cv2.rectangle(frame, (x, y - 35), (x + w, y), color, cv2.FILLED)
        
        # Name text
        cv2.putText(frame, name, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    return frame

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    frame_count = 0
    recognition_results = []
    
    # FPS Calculation
    start_time = time.time()
    fps = 0
    
    print("System Started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        frame_count += 1
        
        # --- Recognition Logic (Skipping Frames) ---
        if frame_count % FRAME_SKIP == 0:
            # Run recognition in a separate call to avoid blocking UI too much?
            # For simplicity in this script, it blocks, but skipping helps.
            recognition_results = recognize_frame(frame)
            
            # Log attendance for found faces
            for res in recognition_results:
                log_attendance(res['name'])

        # --- Draw Overlay ---
        # We reuse the last known recognition_results for frames between checks
        # to prevent flickering, though boxes might lag slightly behind movement.
        
        # Calculate FPS
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = end_time

        frame = draw_overlay(frame, recognition_results, fps)

        cv2.imshow('Face Recognition Attendance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
