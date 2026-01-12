import uvicorn
import os
import sys
# --- CPU OPTIMIZATION FOR TENSORFLOW ---
# Set these BEFORE importing other libraries that might initialize TF
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# ---------------------------------------
from pymongo import MongoClient
import time

def check_mongo():
    print("Checking MongoDB connection...")
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.server_info()
        print("✅ MongoDB is running and accessible.")
        return True
    except Exception as e:
        print("❌ Error: Could not connect to MongoDB.")
        print("Please ensure MongoDB is installed and running on localhost:27017")
        print(f"Details: {e}")
        print("[!] WARNING: MongoDB not found. Application will use IN-MEMORY DUMMY DATABASE.")
        return True # Allow fallback

def main():
    check_mongo() # check but don't stop


    print("Starting Web Application...")
    # Run the uvicorn server
    # We use the string "web_app:app" to allow reloading
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
