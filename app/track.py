from utils import get_name, get_current_time
import uuid

def add_to_dictionary(dictionary, uid):
    """
    args:
        dictionary (dict): The dictionary to update.
        uid (str): The UID to add or update in the dictionary.
    returns:
        tuple: (exists, updated_dictionary)

    This function checks if the UID exists in the dictionary and adds it if not found.
    If the UID is found, it returns True and the existing dictionary.
    """
    track_id = uuid.uuid4().hex[:8]  # Generate a unique track ID
    print(f"[INFO] Generated track ID: {track_id}")
    name = get_name(uid) # Get the name associated with the UID

    if name == 'Unknown':
        print(f"[ERROR] No name found for UID: {uid}")
        return None
    else:
        print(f"[INFO] Name found for UID {uid}: {name}")

    date, time = get_current_time()

    if review_dictionary(dictionary, uid)[0] is None:
        dictionary[track_id] = {
            'uid': uid,
            'name': name,
            'date': date,
            'time': time
        }
        print(f"[INFO] Added {name} with UID {uid} to the dictionary.")
        return False, dictionary
    
    else:
        print(f"[INFO] {name} with UID {uid} already exists in the dictionary.")
        return True, dictionary


def review_dictionary(dictionary, uid):
    """
    args:
        dictionary (dict): The dictionary to search.
        uid (str): The UID to search for.
    returns:
        tuple: (track_id, details) if found, else (None, None)

    This function checks if the UID exists in the dictionary and returns the associated track ID and details.
    """
    name = get_name(uid)

    for track_id, details in dictionary.items():
        if details['name'] == name:
            print(f"[INFO] Found {name} with UID {uid} against track ID {track_id} in the dictionary at {details['date']} {details['time']}.")
            return track_id, details
        
    print(f"[ERROR] {name} with UID {uid} not found in the dictionary.")
    return None, None