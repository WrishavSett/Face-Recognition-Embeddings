import os
import csv
import numpy as np
from elevenlabs import ElevenLabs
import datetime
from dotenv import load_dotenv

load_dotenv()

def get_name(uid):
    name_dict = {
        "TNU2020021100001": "Subhadip Samanta",
        "TNU2020021100002": "Subhodeep Ghosh",
        "TNU2020021100004": "Wrishav Sett",
        "TNU2020021100005": "Sudipta Saha",
        "TNU2020021100006": "Subhajit Paul",
        "TNU2020021100007": "Yuvraj Singh Negi",
        "TNU2020021100009": "Pratap Sinha",
        "TNU2020021100011": "Rajkumar Maity",
        "TNU2020053100001": "Ayan Pramanik",
        "TNU2020053100003": "Rajkumar Roy",
        "TNU2020053100004": "Sayak Mondal",
        "TNU2020053100006": "Srikanta Pramanik",
        "TNU2020053100007": "D Omkar Murty",
        "TNU2020053100009": "",
        "TNU2020053100011": "",
        "TNU2020053100013": "",
        "TNU2020053100014": "Md Zunnurain",
        "TNU2020053100018l": "",
        "TNU2020053100031l": ""
        }
    return name_dict.get(uid, 'Unknown')

def generate_voice(uid):
    client = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        )
    
    text, welcome = create_greeting(uid)
    name = get_name(uid)

    if welcome:
        path = f"./temp/{name}_welcome.mp3"
    else:
        path = f"./temp/{name}_goodbye.mp3"
    if os.path.isfile(path):
        print(f"[INFO] Audio file already exists for UID: {uid}, using existing file.")
        return path

    print(f"[INFO] Generating voice for UID: {uid} with name: {name}")

    if text is None:
        text = "Hello, this is a test of the ElevenLabs text-to-speech service."

    response = client.text_to_speech.convert(
        voice_id="21m00Tcm4TlvDq8ikWAM",
        output_format="mp3_44100_128",
        text=text,
        model_id="eleven_multilingual_v2"
    )

    # Save the audio content to a file
    with open(path, "wb") as f:
        for chunk in response:
            f.write(chunk)
    print(f"[INF0] Audio successfully saved to {path}")
    return path

def get_current_time():
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    return date, time

def create_greeting(uid):
    date, time = get_current_time()

    name = get_name(uid)
    welcome = True

    if time > "08:45:00" and time < "17:45:00":
        greeting = f"Welcome, {name}!"
        welcome = True
    elif time > "17:45:00" and time < "23:59:59":
        greeting = f"Goodbye, {name}!"
        welcome = False

    return greeting, welcome

def play_sound(uid):
    path = generate_voice(uid)
    try:
        from playsound import playsound
        playsound(path)
    except Exception as e:
        print(f"[ERROR] Failed to play sound: {e}")
        raise e
    
# --- Utility Functions ---
def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def write_log(uid, name, log_in, log_out, LOG_FILE):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['UID', 'Name', 'Login Time', 'Logout Time'])
        writer.writerow([uid, name, log_in, log_out])
    print(f"[INFO] Logged {uid}: {log_in} - {log_out}")

def check_and_log_day_end(welcome_dictionary, goodbye_dictionary, LOG_FILE):
    print("[INFO] Running end-of-day logging.")
    logged_uids = set()
    for track_id, entry in welcome_dictionary.items():
        uid = entry['uid']
        name = entry['name']
        log_in = entry['time']
        logout_time = entry['last_seen']
        for goodbye_entry in goodbye_dictionary.values():
            if goodbye_entry['uid'] == uid:
                logout_time = goodbye_entry['last_seen']
                break
        write_log(uid, name, log_in, logout_time, LOG_FILE)
        logged_uids.add(uid)
    print(f"[INFO] Logged {len(logged_uids)} users for logout.")