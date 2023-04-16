import json
import os
from argparse import ArgumentParser
from tqdm import tqdm


def get_playlists(data):
    for plist in data["playlists"]:
        sequence = [song["track_uri"] for song in plist["tracks"]]
        yield sequence


def dump(batch, path):
    print(f"Writing {path}")
    with open(path, "w") as f:
        json.dump({"playlists": batch}, f)
        

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, help="input folder containing original spotify dataset slices", required=True)
    parser.add_argument("--output", type=str, help="output folder", required=True)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--prefix", type=str, default="chunk_")

    args = parser.parse_args()

    assert os.path.isdir(args.input)
    assert os.path.isdir(args.output)

    batch = []
    i = 0
    for fname in tqdm(list(sorted(os.listdir(args.input)))):
        if not (fname.startswith("mpd.slice.") and fname.endswith(".json")):
            continue
            
        fullpath = os.sep.join((args.input, fname))
        with open(fullpath, "r") as f:
            data = json.load(f)
        batch += list(get_playlists(data))
        
        if len(batch) >= args.chunksize:
            out_name = f"{args.prefix}{i}.json"
            out_path = os.path.join(args.output, out_name)
            dump(batch, out_path)
            batch = []
            i += 1
            
    if len(batch) > 0:
        out_name = f"{args.prefix}{i}.json"
        out_path = os.path.join(args.output, out_name)
        dump(batch, out_path)



if __name__ == "__main__":
    main()

