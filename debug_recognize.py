import os
import sys
import numpy as np
from deepface import DeepFace

# Ensure we can import web_app
sys.path.append(os.getcwd())
import web_app

# --- Configuration ---
# 1. PARAMETERS USED FOR DETECTION & RECOGNITION
# These mirror the settings in web_app.py
MODEL_USED = web_app.MODEL_NAME      # "ArcFace"
DETECTOR_BACKEND = "retinaface"      # Used inside process_image for accuracy
METRIC = "cosine"                    # web_app.get_cosine_distance
THRESHOLD = 0.81                     # Hardcoded in web_app.process_image

# Test Image
TEST_IMAGE = "d:\\Atd\\static\\uploads\\2d4fbbca-1315-4056-b28f-0b172f0599cb_WhatsApp Image 2026-01-09 at 4.14.23 PM.jpeg"

def print_graph(val, max_val=1.5, width=40):
    """Prints a simple ASCII bar"""
    chars = int((val / max_val) * width)
    chars = min(chars, width)
    return "[" + "#" * chars + " " * (width - chars) + "]"

def main():
    print("\n" + "="*60)
    print("          FACE RECOGNITION PARAMETERS & GRAPH DEBUG")
    print("="*60)
    
    # 1. Show Parameters
    print(f"[*] Model Name:       {MODEL_USED}")
    print(f"[*] Detector:         {DETECTOR_BACKEND}")
    print(f"[*] Distance Metric:  {METRIC}")
    print(f"[*] Threshold:        {THRESHOLD} (Distances below this are matches)")
    print("-" * 60)

    # 2. Check Image
    if not os.path.exists(TEST_IMAGE):
        print(f"[!] Error: Test image not found at:\n    {TEST_IMAGE}")
        return

    # 3. Load Database
    print("[*] Loading Face Database...")
    web_app.load_embeddings_into_memory()
    print(f"[*] Database Size:    {len(web_app.cached_embeddings)} faces enrolled.")
    print("-" * 60)

    # 4. Process Image
    print(f"[*] Analyzing Image: {os.path.basename(TEST_IMAGE)}")
    try:
        # Detect Faces
        objs = DeepFace.represent(
            img_path=TEST_IMAGE,
            model_name=MODEL_USED,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )

        if not objs:
            print("[!] No faces detected in the image.")
            return

        print(f"[*] Detected {len(objs)} faces. Comparing with database...")

        for i, face in enumerate(objs):
            print(f"\n>>> FACE #{i+1} ANALYSIS")
            target_embedding = face['embedding']
            
            # Compare with ALL known faces
            comparisons = []
            for known in web_app.cached_embeddings:
                dist = web_app.get_cosine_distance(known['embedding'], target_embedding)
                comparisons.append({
                    "name": known['name'],
                    "distance": dist,
                    "is_match": dist < THRESHOLD
                })
            
            # Sort by distance (closest first)
            comparisons.sort(key=lambda x: x['distance'])
            
            # Print Graph (Top 15 results)
            print(f"\n{'MATCH?':<8} {'NAME':<20} {'DIST':<8} {'VISUAL GRAPH (Lower is Better)'}")
            print("-" * 80)
            
            for item in comparisons[:15]: 
                match_mark = "[MATCH]" if item['is_match'] else ""
                graph = print_graph(item['distance'])
                print(f"{match_mark:<8} {item['name']:<20} {item['distance']:.4f}   {graph}")

            if len(comparisons) > 15:
                print(f"... and {len(comparisons) - 15} others.")

    except Exception as e:
        print(f"[!] Analysis Failed: {e}")

if __name__ == "__main__":
    main()
