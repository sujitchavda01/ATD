import sys
import os

# Mocking things to prevent side effects if needed, but web_app seems safe
# modifying path
sys.path.append(os.getcwd())

from web_app import load_embeddings_into_memory, cached_embeddings, MY_DB_FOLDER, MODEL_NAME, DETECTOR_BACKEND

print(f"Testing DB Build for {MODEL_NAME} with {DETECTOR_BACKEND}")
print(f"DB Folder: {MY_DB_FOLDER}")

try:
    load_embeddings_into_memory()
    from web_app import cached_embeddings # re-import to get updated global? 
    # Actually modifying global in module affects import
    
    import web_app
    print(f"Final cached count: {len(web_app.cached_embeddings)}")
    
    if len(web_app.cached_embeddings) == 0:
        print("WARNING: 0 embeddings found. Listing DB folder to debug...")
        for root, dirs, files in os.walk(MY_DB_FOLDER):
            print(f"Root: {root}, Files: {files}")
            
except Exception as e:
    print(f"Error: {e}")
