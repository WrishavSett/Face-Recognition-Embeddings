import os
from elevenlabs import ElevenLabs
import datetime


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
        api_key="sk_e1c736d567dd9acde67a8661ff4eafe217d979d8fa574063"
        )
    
    text = create_greeting(uid)
    name = get_name(uid)

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
    with open(f"../temp/{name}.mp3", "wb") as f:
        for chunk in response:
            f.write(chunk)
    print(f"[INF0] Audio successfully saved to {name}.mp3")
    return f"../temp/{name}.mp3"

def get_current_time():
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    return date, time

def create_greeting(uid):
    date, time = get_current_time()

    name = get_name(uid)

    if time > "08:45:00" and time < "17:45:00":
        greeting = f"Welcome, {name}!"
    elif time > "17:45:00" and time < "23:59:59":
        greeting = f"Goodbye, {name}!"

    return greeting