import os
import pickle
from deepface import DeepFace
import cv2
import numpy as np

MY_DB_FOLDER = "my_db"
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface" # For accurate enrollment

print(f"[*] Starting Debug Enrollment...")
print(f"[*] DB Folder: {os.path.abspath(MY_DB_FOLDER)}")

if not os.path.exists(MY_DB_FOLDER):
    print("ERROR: DB Folder missing")
    exit()

temp_embeddings = []

for root, dirs, files in os.walk(MY_DB_FOLDER):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            full_path = os.path.join(root, file)
            print(f"Processing: {full_path}")
            
            folder_name = os.path.basename(os.path.dirname(full_path))
            if folder_name == "my_db": continue
            
            clean_name = folder_name.split("(")[0].strip() if "(" in folder_name else folder_name
            
            try:
                # Enforce use of RetinaFace for Enrollment 
                objs = DeepFace.represent(
                    img_path=full_path,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=True 
                )
                
                if objs:
                    # FIX: Pick the LARGEST face, not just the first one.
                    # This prevents enrolling background bystanders in the base photo.
                    best_face = max(objs, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
                    
                    embedding = best_face['embedding']
                    temp_embeddings.append({"name": clean_name, "embedding": embedding})
                    print(f"   -> Enrolled: {clean_name} (Size: {best_face['facial_area']['w']}x{best_face['facial_area']['h']})")
                else:
                    print("   -> No embedding returned.")
                    
            except Exception as e:
                print(f"   [!] Error enrolling {file}: {e}")
                # Try fallback
                try:
                    print("     -> Retrying with SSD...")
                    objs = DeepFace.represent(
                        img_path=full_path,
                        model_name=MODEL_NAME,
                        detector_backend="ssd",
                        enforce_detection=True 
                    )
                    if objs:
                         temp_embeddings.append({"name": clean_name, "embedding": objs[0]['embedding']})
                         print(f"   -> Enrolled (SSD): {clean_name}")
                except Exception as ex:
                    print(f"     -> Failed SSD too: {ex}")

print(f"[*] Total Enrolled: {len(temp_embeddings)}")

if len(temp_embeddings) > 0:
    db_file_name = f"embeddings_{MODEL_NAME}_master.pkl"
    custom_db_path = os.path.join(MY_DB_FOLDER, db_file_name)
    with open(custom_db_path, 'wb') as f:
        pickle.dump(temp_embeddings, f)
    print(f"[*] Saved to {custom_db_path}")
else:
    print("[!] 0 Enrollments!")
