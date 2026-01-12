import os
import cv2
import shutil
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, Request, File, UploadFile, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pymongo import MongoClient
from deepface import DeepFace
import pandas as pd
import numpy as np
import threading
import queue
import time
import concurrent.futures

# --- Configuration ---
DB_URI = "mongodb://localhost:27017/"
DB_NAME = "face_attendance_db"
MY_DB_FOLDER = "my_db"
STATIC_DIR = "static"
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
MODEL_NAME = "ArcFace" 
# CRITICAL: Back to RetinaFace for DETECTING EVERYONE. 
# We will optimize speed by resizing images in the worker.
DETECTOR_BACKEND = "retinaface" 
FRAME_SKIP = 5  
DB_PICKLE_FILE = f"embeddings_{MODEL_NAME}_Retina_master.pkl" 

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MY_DB_FOLDER, exist_ok=True)

# --- App Setup ---
app = FastAPI(title="Face Attendance System")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/dataset", StaticFiles(directory=MY_DB_FOLDER), name="dataset") # Serve dataset images
templates = Jinja2Templates(directory="templates")

# --- Database Setup ---
try:
    client = MongoClient(DB_URI, serverSelectionTimeoutMS=2000)
    client.server_info() # Trigger connection check
    print(f"[*] Connected to MongoDB at {DB_URI}")
    db = client[DB_NAME]
except Exception as e:
    print(f"[!] MongoDB Connection Failed: {e}")
    print("[*] Switching to In-Memory Dummy Database (NO PERSISTENCE)")
    from dummy_db import MockClient, sync_users_from_disk
    client = MockClient(DB_URI)
    db = client[DB_NAME]
    
    # Pre-populate Mock DB from disk so UI isn't empty
    # We do this after assigning users_collection below
    
users_collection = db["users"]
attendance_collection = db["attendance"]

# If using Mock DB, sync users now
if isinstance(client, (type(None), object)) and "MockClient" in str(type(client)): # Robust check or just flag
     pass # handled below

try:
    from dummy_db import MockClient
    if isinstance(client, MockClient):
        sync_users_from_disk(users_collection, MY_DB_FOLDER)
except:
    pass


# --- Memory Cache for Real-Time Video ---
cached_embeddings = [] # List of {"name": str, "embedding": list}
last_db_modification = 0

# --- Worker Queue System ---
job_queue = queue.Queue()
batch_status_store = {} # {batch_id: {status: 'pending'|'processing'|'completed', total: int, processed: int, result_id: str}}

def fix_orientation(file_path):
    """Fixes image rotation based on EXIF data."""
    try:
        from PIL import Image, ExifTags
        image = Image.open(file_path)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = image._getexif()
        if exif is not None and orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
            image.save(file_path)
            # print(f"[*] Fixed orientation: {os.path.basename(file_path)}")
    except:
        pass

def attendance_worker():
    """
    Background worker that constantly listens for new batches of images to process.
    Optimized for FASTEST throughput using ThreadPoolExecutor for concurrent I/O & Processing.
    """
    print("[*] Worker Thread Started... Waiting for jobs.")
    
    while True:
        try:
            # Get job from queue (blocking)
            batch_id, file_list = job_queue.get()
            
            print(f"[*] Worker picked up batch {batch_id} with {len(file_list)} files.")
            batch_status_store[batch_id]['status'] = 'processing'
            
            start_time = time.time() # Start Timer
            
            all_detected_names = set()
            detection_scores = {} # Dictionary to store scores {name: 98.5}
            processed_count = 0
            processed_files_info = []
            
            # Helper for single file processing
            def process_item(item):
                f_path, is_vid, f_name = item
                names_and_scores = []
                try:
                    # 1. Fix Orientation (Crucial for phone uploads)
                    if not is_vid:
                        fix_orientation(f_path)
                        
                        # 2. Resize to Optimize RetinaFace Speed
                        # 1024px: The Sweet Spot. 
                        # - Detects group faces correctly.
                        # - Runs 2x faster than 1400px.
                        # - Direct memory pass (no I/O) for speed.
                        img = cv2.imread(f_path)
                        h, w = img.shape[:2]
                        MAX_DIM = 1024 
                        if max(h, w) > MAX_DIM:
                            scale = MAX_DIM / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            # INTER_AREA for sharpness
                            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                        # Skip writing to disk! Save I/O time.
                        # cv2.imwrite(f_path, img) 
                        
                        # process_image now returns list of dicts: [{'name': 'Bob', 'accuracy': 80.5}]
                        # Pass the numpy image directly to avoid re-reading
                        names_and_scores = process_image(f_path, img_input=img) 
                    else:
                        # process_video still returns list of strings for now, wrapper it
                        # TODO: update video to return scores later
                        names_only = process_video(f_path)
                        names_and_scores = [{"name": n, "accuracy": 0.0} for n in names_only]

                except Exception as e:
                    print(f"[!] Error in worker for {f_name}: {e}")
                
                return names_and_scores, f_name

            # PARALLEL EXECUTION
            # High Performance Mode
            # 8 workers to saturate CPU cores.
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(process_item, item): item for item in file_list}
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        detected_objs, fname = future.result()
                        
                        for obj in detected_objs:
                            name = obj['name']
                            score = obj['accuracy']
                            all_detected_names.add(name)
                            
                            # Store the highest accuracy encountered for this person
                            if name not in detection_scores or score > detection_scores[name]:
                                detection_scores[name] = score
                                
                        processed_files_info.append(f"/static/uploads/{fname}")
                    except Exception as e:
                        print(f"[!] Job failed: {e}")
                    
                    # Update Progress
                    processed_count += 1
                    batch_status_store[batch_id]['processed'] = processed_count
            
            # Create Result Record
            all_users = [u['name'] for u in get_all_users()]
            absent = list(set(all_users) - all_detected_names)
            
            end_time = time.time()
            duration = round(end_time - start_time, 2)

            record = {
                "timestamp": datetime.now(),
                "file_path": processed_files_info[0] if processed_files_info else "", 
                "file_count": len(file_list),
                "type": "batch" if len(file_list) > 1 else ("video" if file_list[0][1] else "image"),
                "detected_count": len(all_detected_names),
                "total_users": len(all_users),
                "detected_names": list(all_detected_names),
                "scores": detection_scores, # SAVE SCORES
                "absent_names": absent,
                "accuracy_model": MODEL_NAME,
                "processing_time_sec": duration
            }
            
            result = attendance_collection.insert_one(record)
            
            # Mark Complete
            batch_status_store[batch_id]['status'] = 'completed'
            batch_status_store[batch_id]['result_id'] = str(result.inserted_id)
            print(f"[*] Batch {batch_id} Finished.")
            
            job_queue.task_done()
            
        except Exception as e:
            print(f"[!] Worker Global Error: {e}")

# Start Worker Thread
threading.Thread(target=attendance_worker, daemon=True).start()

def load_embeddings_into_memory():
    """
    Manually builds a lightweight face database.
    Uses 'retinaface' for ACCURACY during enrollment (ensures faces are actually found).
    """
    global cached_embeddings, last_db_modification
    import pickle
    
    # We use a generic name because the detector for enrollment (RetinaFace) 
    # might differ from video detector (YuNet), but embeddings are compatible 
    # as long as the MODEL (GhostFaceNet) is the same.
    db_file_name = DB_PICKLE_FILE
    
    # Use ABSOLUTE PATH to ensure we find it regardless of CWD
    custom_db_path = os.path.join(os.path.abspath(MY_DB_FOLDER), db_file_name)
    
    print(f"[*] Looking for DB at: {custom_db_path}")

    # 1. Try to load existing DB
    if os.path.exists(custom_db_path):
        if not cached_embeddings:
            print(f"[*] Loading Master Face DB: {db_file_name}")
            try:
                with open(custom_db_path, 'rb') as f:
                    # FIX: Force load into temporary variable first
                    loaded_data = pickle.load(f)
                    cached_embeddings = loaded_data # Assign to global
                print(f"[*] Loaded {len(cached_embeddings)} faces from disk. SAMPLE: {cached_embeddings[0]['name'] if cached_embeddings else 'EMPTY'}")
                return
            except Exception as e:
                print(f"[!] DB Corrupt, rebuilding: {e}")
        else:
            print(f"[*] Memory already holds {len(cached_embeddings)} faces.")
            return
    else:
        print(f"[!] DB File Missing: {custom_db_path}")

    # 2. Rebuild Database (Iterate folders)
    print(f"[*] Building Master Face Database (High Accuracy Mode)...")
    temp_embeddings = []
    
    for root, dirs, files in os.walk(MY_DB_FOLDER):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                full_path = os.path.join(root, file)
                
                folder_name = os.path.basename(os.path.dirname(full_path))
                if folder_name == "my_db": continue
                
                clean_name = folder_name.split("(")[0].strip() if "(" in folder_name else folder_name
                
                try:
                    # Enforce use of RetinaFace for Enrollment to ensure we capture the face
                    objs = DeepFace.represent(
                        img_path=full_path,
                        model_name=MODEL_NAME,
                        detector_backend="retinaface", # FORCE RETINAFACE FOR REGISTRATION
                        enforce_detection=True # We WANT to fail if no face found in training data
                    )
                    
                    if objs:
                        # PICK THE LARGEST FACE for enrollment to avoid bystanders
                        best_face = max(objs, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
                        embedding = best_face['embedding']
                        temp_embeddings.append({"name": clean_name, "embedding": embedding})
                        print(f"   -> Enrolled: {clean_name}")
                        
                except Exception as e:
                     # Fallback to SSD if RetinaFace fails (rare)
                     try:
                        objs = DeepFace.represent(
                            img_path=full_path, 
                            model_name=MODEL_NAME, 
                            detector_backend="ssd", 
                            enforce_detection=True
                        )
                        if objs:
                            temp_embeddings.append({"name": clean_name, "embedding": objs[0]['embedding']})
                            print(f"   -> Enrolled (SSD): {clean_name}")
                     except:
                        print(f"   [!] Failed to enroll {clean_name} (No face found).")
    
    # 3. Save to disk
    if temp_embeddings:
        try:
            with open(custom_db_path, 'wb') as f:
                pickle.dump(temp_embeddings, f)
            print(f"[*] Database Saved! Total Faces: {len(temp_embeddings)}")
            cached_embeddings = temp_embeddings
        except Exception as e:
            print(f"[!] Failed to save DB: {e}")
    else:
        print("[!] CRITICAL: No faces were enrolled. Check your images.")

def get_cosine_distance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def calculate_accuracy_percentage(distance, threshold):
    """
    Maps the cosine distance to a user-friendly accuracy percentage.
    ArcFace distances:
    < 0.30 : Excellent Match (90-100%)
    < 0.50 : Good Match (70-90%)
    < 0.68 : Acceptable Match (50-70%)
    """
    if distance <= 0: return 100.0
    if distance >= threshold: return 0.0 # Or map to < 50%
    
    # Piecewise mapping for better user perception
    if distance < 0.35: # Widened excellent range
        # Map 0.0-0.35 to 100-90
        return 100.0 - (distance / 0.35) * 10.0
    else:
        # Map 0.35-threshold to 90-50
        range_span = threshold - 0.35
        dist_in_range = distance - 0.35
        fraction = dist_in_range / range_span
        # We want to go from 90 down to 50 (drop of 40)
        return 90.0 - (fraction * 40.0)

# --- Helper Functions ---

def get_all_users():
    """Retrieve all registered users from MongoDB."""
    return list(users_collection.find({}, {"_id": 0, "name": 1, "division": 1, "photo_url": 1, "folder_name": 1}))

def process_image(file_path, img_input=None):
    """
    Process a single image path for face recognition.
    Returns list of detected names.
    Annotates the image with bounding boxes and accuracy.
    OPTIMIZED: Uses In-Memory Cache + YuNet + GhostFaceNet for <1s processing.
    """
    detected_names = []
    results_list = []
    print(f"[*] Processing Image: {file_path}")
    
    # Ensure cache is loaded
    load_embeddings_into_memory()
    
    # Pre-build model to avoid race conditions in threads
    try:
        DeepFace.build_model(MODEL_NAME)
    except: pass
    
    # Read image for annotation optimization (Avoid double read)
    if img_input is not None:
        img_cv2 = img_input
    else:
        img_cv2 = cv2.imread(file_path)
    
    try:
        # 1. Detect & Represent all faces in the image (Fastest way)
        # Using RetinaFace + 1024px Resize (Balanced Speed/Accuracy)
        face_objs = DeepFace.represent(
            img_path=img_cv2,
            model_name=MODEL_NAME,
            detector_backend="retinaface", 
            enforce_detection=False,
            align=True
        )
        
        print(f"[*] Faces detected in image: {len(face_objs)}")
        
        for i, face in enumerate(face_objs):
            target_embedding = face['embedding']
            area = face['facial_area']
            x, y, w, h = area['x'], area['y'], area['w'], area['h'] # Safe access
            
            # 2. Compare against MEMORY CACHE (No Disk I/O)
            best_distance = 1.0
            best_name = "Unknown"
            
            if not cached_embeddings:
                print("    [!] CRITICAL: Cache is empty during recognition!")
            
            for known in cached_embeddings:
                 # GhostFaceNet + Cosine
                 try:
                     dist = get_cosine_distance(known['embedding'], target_embedding)
                     if dist < best_distance:
                         best_distance = dist
                         best_name = known['name']
                 except Exception as e:
                     print(f"Error calculating distance: {e}")
            
            # Match logic
            threshold = 0.75 # Relaxed slightly for group photos (Standard is 0.68)
            
            if best_distance < threshold:
                # MATCH FOUND
                # Calc accuracy score using improved mapping
                accuracy_score = calculate_accuracy_percentage(best_distance, threshold)
                
                print(f"    -> Face {i+1} identified as: {best_name} (Dist: {best_distance:.4f}) -> {accuracy_score:.1f}%")
                
                results_list.append({"name": best_name, "accuracy": round(accuracy_score, 1)})
                detected_names.append(best_name) # Keep for counting
                
                # Annotate
                if img_cv2 is not None:
                    cv2.rectangle(img_cv2, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    text = f"{best_name} ({accuracy_score:.0f}%)"
                    cv2.putText(img_cv2, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # UNKNOWN
                print(f"    -> Face {i+1}: Unknown. Best: {best_name} (Dist: {best_distance:.4f})") 
                if img_cv2 is not None:
                    cv2.rectangle(img_cv2, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    text = f"Unknown"
                    cv2.putText(img_cv2, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    except Exception as e:
        print(f"[!] Error processing image: {e}")
        
    # --- Statistics ---
    total_faces = len(face_objs) if 'face_objs' in locals() else 0
    recognized_faces = len(results_list)
    accuracy_percent = (recognized_faces / total_faces * 100) if total_faces > 0 else 0
    
    print(f"[*] RESULTS: Found {total_faces} faces, Recognized {recognized_faces}.")
    print(f"[*] OVERALL ACCURACY: {accuracy_percent:.1f}%")

    # Save the annotated image back to disk
    if img_cv2 is not None:
        cv2.imwrite(file_path, img_cv2)
        
    return results_list # Return list of objects

def process_video(file_path):
    """
    Process a video file. Skips frames for optimization.
    Returns unique list of detected names.
    """
    print(f"[*] Processing Video: {file_path}")
    cap = cv2.VideoCapture(file_path)
    detected_names = set()
    frame_count = 0
    
    # VIDEO OPTIMIZATION: Frame Skip + Resize
    # To prevent CPU pegging at 100%, we significantly reduce workload.
    VIDEO_PROCESS_WIDTH = 1000 # HD Quality
    FRAME_SKIP = 3 # High frequency scan
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # Resize the frame for processing (Optimization)
        h, w = frame.shape[:2]
        if w > VIDEO_PROCESS_WIDTH:
            scale = VIDEO_PROCESS_WIDTH / w
            new_h = int(h * scale)
            small_frame = cv2.resize(frame, (VIDEO_PROCESS_WIDTH, new_h))
        else:
            small_frame = frame
            
        # We need to save frame to disk for DeepFace.find or pass numpy array
        # Passing numpy array (frame) directly works in recent DeepFace versions
        try:
           # OPTIMIZATION: Use Memory Cache instead of DeepFace.find for 100x speedup
           load_embeddings_into_memory()
           
           if not cached_embeddings:
               # Fallback if cache empty (shouldn't happen if users exist)
               pass
           else:
               # 1. Detect & Represent faces in the current frame
               # UPDATED: Use RetinaFace for Video too
               # We reduced resolution to 800px to compensate.
               frame_objs = DeepFace.represent(
                   img_path=small_frame,
                   model_name=MODEL_NAME,
                   detector_backend="retinaface", 
                   enforce_detection=False,
                   align=True
               )
               
               for obj in frame_objs:
                   target_embedding = obj['embedding']
                   
                   # 2. Manual Search in Memory (Milliseconds)
                   best_distance = 1.0
                   best_name = "Unknown"
                   
                   for known in cached_embeddings:
                       dist = get_cosine_distance(known['embedding'], target_embedding)
                       if dist < best_distance:
                           best_distance = dist
                           best_name = known['name']
                   
                   # 3. Threshold Check
                   # ArcFace video frames might be blurry, so we use a generous threshold.
                   threshold = 0.81
                   
                   if best_distance < threshold: 
                       if best_name not in detected_names:
                            print(f"    [Frame {frame_count}] MATCH: {best_name} | Dist: {best_distance:.4f}")
                            detected_names.add(best_name)
                   else:
                       # Debug low confidence matches
                       pass
                       # print(f"    [Frame {frame_count}] Unknown. Best: {best_name} (Dist: {best_distance:.4f})")

        except Exception as e:
            # print(f"Frame processing error: {e}")
            pass
            
    cap.release()
    print(f"[*] Video processing complete. Found: {list(detected_names)}")
    return list(detected_names)

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    total_users = users_collection.count_documents({})
    total_sessions = attendance_collection.count_documents({})
    
    # Get recent attendance logs
    recent_logs = list(attendance_collection.find().sort("timestamp", -1).limit(5))
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "total_users": total_users,
        "total_sessions": total_sessions,
        "recent_logs": recent_logs
    })

@app.get("/users", response_class=HTMLResponse)
async def list_users(request: Request):
    users = get_all_users()
    return templates.TemplateResponse("users.html", {"request": request, "users": users})

@app.get("/add_user", response_class=HTMLResponse)
async def add_user_form(request: Request):
    return templates.TemplateResponse("add_user.html", {"request": request})

@app.post("/add_user")
async def add_user_submit(request: Request, name: str = Form(...), division: str = Form(...), photo: UploadFile = File(...)):
    # 1. Create folder in my_db
    # Store as "Name (Div)" to avoid collisions and effectively store division
    folder_name = f"{name} ({division})"
    user_folder = os.path.join(MY_DB_FOLDER, folder_name)
    os.makedirs(user_folder, exist_ok=True)
    
    # 2. Save file
    file_ext = photo.filename.split(".")[-1]
    filename = f"{name}_base.{file_ext}"
    file_path = os.path.join(user_folder, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(photo.file, buffer)
        
    print(f"[+] New User Registered: {name} | Division: {division}")
    print(f"    Photo saved at: {file_path}")

    # --- DUPLICATE CHECK ---
    # Check if this face already exists under a different name
    try:
        dup_check = DeepFace.find(
            img_path=file_path,
            db_path=MY_DB_FOLDER,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric="cosine",
            enforce_detection=False,
            silent=True
        )
        for df in dup_check:
            if not df.empty:
                match = df.iloc[0]
                distance = match.get('distance', 0.0)
                source_path = match['identity']
                
                # Check strict distance (e.g., < 0.30 means it's definitely the same person)
                # Ensure we aren't matching the file we just saved (check folder name)
                matched_folder = os.path.basename(os.path.dirname(source_path))
                
                if distance < 0.40 and matched_folder != folder_name:
                    print(f"[!] DUPLICATE DETECTED: Matches {matched_folder} (Dist: {distance:.4f})")
                    # Cleanup the new file/folder
                    shutil.rmtree(user_folder) 
                    return HTMLResponse(
                        content=f"""
                        <script>
                            alert('Registration Failed: This person is already registered as "{matched_folder}"!');
                            window.history.back();
                        </script>
                        """, status_code=200
                    )
    except Exception as e:
        print(f"[!] Warning: Duplicate check failed {e}")
    # -----------------------

    # 3. Add to MongoDB
    # Clean old DeepFace representations to force reload of this new user
    pkl_path = os.path.join(MY_DB_FOLDER, f"representations_{MODEL_NAME}.pkl")
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
        
    user_doc = {
        "name": name,
        "division": division,
        "folder_name": folder_name,
        "photo_url": f"/dataset/{folder_name}/{filename}", 
        "created_at": datetime.now()
    }
    # Update based on stored folder name key if possible, but name+div is unique key here essentially
    users_collection.update_one({"name": name, "division": division}, {"$set": user_doc}, upsert=True)
    
    return RedirectResponse(url="/users", status_code=303)

@app.get("/attendance", response_class=HTMLResponse)
async def attendance_form(request: Request):
    return templates.TemplateResponse("attendance.html", {"request": request})

@app.post("/attendance")
async def attendance_process(request: Request, files: List[UploadFile] = File(...)):
    # 1. Save all files immediately
    saved_files = []
    for file in files:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov'))
        saved_files.append((file_path, is_video, unique_filename))
        
    # 2. Create Batch Job
    batch_id = str(uuid.uuid4())
    batch_status_store[batch_id] = {
        'status': 'pending',
        'total': len(saved_files),
        'processed': 0,
        'result_id': None
    }
    
    # 3. Enqueue
    job_queue.put((batch_id, saved_files))
    
    # 4. Redirect to Processing Page
    return templates.TemplateResponse("processing.html", {"request": request, "batch_id": batch_id})

@app.get("/attendance/status/{batch_id}")
async def get_batch_status(batch_id: str):
    status = batch_status_store.get(batch_id)
    if not status:
        return JSONResponse({"status": "error", "message": "Batch ID not found"}, status_code=404)
    return JSONResponse(status)

@app.get("/attendance/result/{record_id}", response_class=HTMLResponse)
async def attendance_result(request: Request, record_id: str):
    from bson import ObjectId
    record = attendance_collection.find_one({"_id": ObjectId(record_id)})
    return templates.TemplateResponse("result.html", {"request": request, "record": record})

@app.get("/user/delete/{folder_name}")
async def delete_user(request: Request, folder_name: str):
    # 1. Delete from MongoDB
    users_collection.delete_one({"folder_name": folder_name})
    
    # 2. Delete Folder
    folder_path = os.path.join(MY_DB_FOLDER, folder_name)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        
    # 3. Clear Cache to force reload
    global cached_embeddings
    cached_embeddings = [] 
    
    # 4. Remove pickle file to ensure freshness
    pkl_path = os.path.join(MY_DB_FOLDER, DB_PICKLE_FILE)
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
        
    return RedirectResponse(url="/users", status_code=303)

@app.get("/user/edit/{folder_name}", response_class=HTMLResponse)
async def edit_user_form(request: Request, folder_name: str):
    user = users_collection.find_one({"folder_name": folder_name})
    if not user:
        return RedirectResponse(url="/users", status_code=303)
    return templates.TemplateResponse("edit_user.html", {"request": request, "user": user})

@app.post("/user/edit/{folder_name}")
async def edit_user_submit(request: Request, folder_name: str, name: str = Form(...), division: str = Form(...), photo: UploadFile = File(None)):
    global cached_embeddings # Declare global at start of function
    
    user = users_collection.find_one({"folder_name": folder_name})
    if not user:
        return RedirectResponse(url="/users", status_code=303)

    new_folder_name = f"{name} ({division})"
    old_folder_path = os.path.join(MY_DB_FOLDER, folder_name)
    new_folder_path = os.path.join(MY_DB_FOLDER, new_folder_name)
    
    # Update Data
    update_data = {
        "name": name, 
        "division": division, 
        "folder_name": new_folder_name,
        "photo_url": user.get('photo_url', '') # keep old unless changed
    }

    # 1. Rename Folder if changed
    if folder_name != new_folder_name:
        if os.path.exists(old_folder_path):
            os.rename(old_folder_path, new_folder_path)
            # Update photo url path if it contains the old folder name
            if update_data['photo_url'] and folder_name in update_data['photo_url']:
                 # Use simple string replace, assuming standard format
                 # safer to reconstruct if possible, but replace is okay here
                 update_data['photo_url'] = update_data['photo_url'].replace(folder_name, new_folder_name)

    # 2. Handle New Photo if provided
    if photo and photo.filename:
        # Save new photo
        file_ext = photo.filename.split(".")[-1]
        filename = f"{name}_base.{file_ext}"
        
        # Ensure new folder exists (if renamed, it exists, if not, it exists)
        final_folder_path = new_folder_path
        os.makedirs(final_folder_path, exist_ok=True) # Safety
        
        file_path = os.path.join(final_folder_path, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)
            
        update_data["photo_url"] = f"/dataset/{new_folder_name}/{filename}"
        
        # Clear cache/pickle
        cached_embeddings = []
        pkl_path = os.path.join(MY_DB_FOLDER, DB_PICKLE_FILE)
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
    elif folder_name != new_folder_name:
         # Name changed but no new photo => We still need to invalidate cache 
         # because the "name" associated with embedding in memory is old.
         cached_embeddings = []
         pkl_path = os.path.join(MY_DB_FOLDER, DB_PICKLE_FILE)
         if os.path.exists(pkl_path):
            os.remove(pkl_path)

    
    # 3. Update DB
    # We use update_one with upsert=True based on new criteria or just replace
    # To avoid complications with keys, we'll update the document identified by _id if we had it, 
    # but since we don't pass _id easily, we'll delete and re-insert or update based on specific unique data.
    # However, 'folder_name' was unique.
    
    users_collection.delete_one({"folder_name": folder_name}) 
    users_collection.insert_one(update_data)

    return RedirectResponse(url="/users", status_code=303)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
