from dotenv import load_dotenv
load_dotenv()

from auth import Auth
import os
import requests
import json
import time
from tqdm import tqdm

songs = json.load(open("../models/all_songs.json"))["songs"]
def batched_gen(lst, bs):
    i = 0
    while i < len(lst):
        yield lst[i:i+bs]
        i += bs


BUFFER_SIZE = 50*500
buffer = []

chunks = [os.path.join("../data/songs_meta", f) for f in os.listdir("./data") if "chunk" in f]
idx = len(chunks)
fetched_songs = []
for c in chunks:
    chunk = json.load(open(c))
    for song in chunk:
        fetched_songs.append(song["uri"])

print("Number of songs:", len(songs))
songs = list(set(songs).difference(fetched_songs))
print("Number of fetched songs", len(fetched_songs))
print("Number of songs to be fetched", len(songs))


t = Auth()

def safe_dict_field(data, *keys):
    for k in keys:
        if not isinstance(data, dict) or k not in data:
            return None
        data = data[k]
    return data

def fetch_song_data(ids):
    global t
    url = f"https://api.spotify.com/v1/tracks?ids={'%2C'.join(ids)}"
    r = requests.get(
        url=url,
        headers={
            "Authorization": f"Bearer {t.token}"
        }
    )
    if r.status_code == 429:
        amount = int(r.headers["Retry-After"])
        print("Sleeping", amount)
        time.sleep(amount)
    r.raise_for_status()
    return r.json()


def fetch_song_features(ids):
    global t
    url = f"https://api.spotify.com/v1/audio-features?ids={'%2C'.join(ids)}"
    r = requests.get(
        url=url,
        headers={
            "Authorization": f"Bearer {t.token}"
        }
    )
    if r.status_code == 429:
        amount = int(r.headers["Retry-After"])
        print("Sleeping", amount)
        time.sleep(amount)
    else:
        r.raise_for_status()
    return r.json()




for sgs in tqdm(batched_gen(songs, 50), total=len(songs)//50):
    ids = [s.split(":")[-1] for s in sgs]
    try:
        data = fetch_song_data(ids)
        features = fetch_song_features(ids)
        if len(data["tracks"]) != len(ids):
            print(data)
            break
        info = [
            {
                "uri": song["uri"],
                "release": safe_dict_field(song, "album", "release_date"),
                "genre": safe_dict_field(song, "album", "genres"),
                "duration": safe_dict_field(song, "duration_ms"),
                "name": safe_dict_field(song, "name"),
                "acousticness": safe_dict_field(feat, "acousticness"),
                "danceability": safe_dict_field(feat, "danceability"),
                "energy": safe_dict_field(feat, "energy"),
                "instrumentalness": safe_dict_field(feat, "instrumentalness"),
                "key": safe_dict_field(feat, "key"),
                "liveness": safe_dict_field(feat, "liveness"),
                "loudness": safe_dict_field(feat, "loudness"),
                "mode": safe_dict_field(feat, "mode"),
                "speechiness": safe_dict_field(feat, "speechiness"),
                "tempo": safe_dict_field(feat, "tempo"),
                "time_signature": safe_dict_field(feat, "time_signature"),
                "valence": safe_dict_field(feat, "valence")
            }
            for song, feat in zip(data["tracks"], features["audio_features"])
            if song is not None and feat["uri"] == song["uri"]
        ]
        if len(info) != len(ids):
            missing = set(sgs).difference([i["uri"] for i in info])
            print("Missing songs", missing)
        buffer += info
        if len(buffer) > BUFFER_SIZE:
            with open(f"./data/chunk_{idx}.json", "w") as f:
                json.dump(buffer, f)
            buffer = []
            idx += 1
    except Exception as e:
        print(e)
    
with open(f"./data/chunk_{idx}.json", "w") as f:
    json.dump(buffer, f)
