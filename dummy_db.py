from bson import ObjectId
import os
import datetime

class MockCursor:
    def __init__(self, data):
        self.data = list(data)  # Make a copy

    def sort(self, key, direction=-1):
        reverse = (direction == -1)
        self.data.sort(key=lambda x: x.get(key, ""), reverse=reverse)
        return self

    def limit(self, n):
        self.data = self.data[:n]
        return self

    def __iter__(self):
        return iter(self.data)
    
    def __list__(self):
        return self.data
    
    def __getitem__(self, index):
        return self.data[index]

class MockResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id

class MockCollection:
    def __init__(self, name, db=None):
        self.name = name
        self.data = [] # List of dicts
        self.db = db

    def insert_one(self, document):
        if "_id" not in document:
            document["_id"] = ObjectId()
        # Deep copy might be better but shallow copy is usually enough for simple dicts
        self.data.append(document.copy())
        return MockResult(document["_id"])

    def find(self, query=None, projection=None):
        results = []
        if query is None:
            query = {}
            
        for doc in self.data:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                results.append(doc)
        
        # Projection handling
        if projection:
            projected_results = []
            exclude_id = projection.get("_id") == 0
            
            # check if it is an inclusion projection (all 1s) or exclusion (all 0s)
            # Mixed is not allowed in mongo except for _id suppression
            keys_to_include = [k for k, v in projection.items() if v == 1 and k != "_id"]
            keys_to_exclude = [k for k, v in projection.items() if v == 0 and k != "_id"]
            
            for doc in results:
                new_doc = {}
                if keys_to_include:
                    # Inclusion mode
                    if not exclude_id:
                        new_doc["_id"] = doc.get("_id")
                    for k in keys_to_include:
                        if k in doc:
                            new_doc[k] = doc[k]
                elif keys_to_exclude:
                    # Exclusion mode
                    new_doc = doc.copy()
                    if exclude_id:
                        new_doc.pop("_id", None)
                    for k in keys_to_exclude:
                        new_doc.pop(k, None)
                else:
                    # Projection was just {"_id": 0} or empty
                    new_doc = doc.copy()
                    if exclude_id:
                        new_doc.pop("_id", None)
                        
                projected_results.append(new_doc)
            return MockCursor(projected_results)
        
        return MockCursor(results)

    def find_one(self, query=None):
        cursor = self.find(query)
        if len(cursor.data) > 0:
            return cursor.data[0]
        return None

    def update_one(self, query, update, upsert=False):
        doc = self.find_one(query)
        if doc:
            # Modify the actual document in self.data
            # find_one returns a copy if I implemented it that way, 
            # but let's make sure we find the index of the original doc
            for i, d in enumerate(self.data):
                if d.get("_id") == doc.get("_id"):
                    # Apply update
                    if "$set" in update:
                        self.data[i].update(update["$set"])
                    break
        elif upsert:
            new_doc = query.copy()
            if "$set" in update:
                new_doc.update(update["$set"])
            self.insert_one(new_doc)
            
    def delete_one(self, query):
        doc = self.find_one(query)
        if doc:
            self.data = [d for d in self.data if d.get("_id") != doc.get("_id")]

    def count_documents(self, query):
        return len(self.find(query).data)

class MockDatabase:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection(name, self)
        return self.collections[name]

class MockClient:
    def __init__(self, uri=None, serverSelectionTimeoutMS=None):
        self.db = MockDatabase()
    
    def __getitem__(self, name):
        return self.db
    
    def server_info(self):
        return {"version": "Mock DB"}

def sync_users_from_disk(users_collection, my_db_folder):
    """
    Scans the MY_DB_FOLDER and populates the users_collection
    if the collection is empty.
    """
    if users_collection.count_documents({}) > 0:
        return

    print("[*] Mock DB: Syncing users from disk...")
    if not os.path.exists(my_db_folder):
        return

    for item in os.listdir(my_db_folder):
        item_path = os.path.join(my_db_folder, item)
        if os.path.isdir(item_path) and item != "my_db":
            # Parsing "Name (Division)"
            name = item
            division = "General"
            
            if "(" in item and item.endswith(")"):
                parts = item.split("(")
                name = parts[0].strip()
                division = parts[1][:-1].strip()
            
            # Find a photo
            photo_url = ""
            for f in os.listdir(item_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                     photo_url = f"/dataset/{item}/{f}"
                     break
            
            user_doc = {
                "name": name,
                "division": division,
                "folder_name": item,
                "photo_url": photo_url,
                "created_at": datetime.datetime.now()
            }
            users_collection.insert_one(user_doc)
    print(f"[*] Mock DB: Loaded {users_collection.count_documents({})} users.")
    
