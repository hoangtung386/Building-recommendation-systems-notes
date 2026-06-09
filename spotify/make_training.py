#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Given Json playlist files makes training data.
"""

import glob
import json
import os
from typing import Any, Dict, Tuple

from absl import app
from absl import flags
from absl import logging
import numpy as np

import input_pipeline

FLAGS = flags.FLAGS
_PLAYLISTS = flags.DEFINE_string("playlists", None, "Playlist json glob.")
_DICTIONARY_PATH = flags.DEFINE_string("dictionaries", "data/dictionaries", "Dictionary path.")
_OUTPUT_PATH = flags.DEFINE_string("output", "data/training", "Output path.")
_TOP_K = flags.DEFINE_integer("topk", 5, "Top K tracks to use as context.")
_MIN_NEXT = flags.DEFINE_integer("min_next", 10, "Min number of tracks.")

flags.mark_flag_as_required("playlists")


def main(argv):
    """Main function."""
    del argv

    playlist_files = glob.glob(_PLAYLISTS.value)

    track_uri_dict = input_pipeline.load_dict(_DICTIONARY_PATH.value, "track_uri_dict.json")
    print("%d tracks loaded" % len(track_uri_dict))
    artist_uri_dict = input_pipeline.load_dict(_DICTIONARY_PATH.value, "artist_uri_dict.json")
    print("%d artists loaded" % len(artist_uri_dict))
    album_uri_dict = input_pipeline.load_dict(_DICTIONARY_PATH.value, "album_uri_dict.json")
    print("%d albums loaded" % len(album_uri_dict))
    topk = _TOP_K.value
    min_next = _MIN_NEXT.value
    print("Filtering out playlists with less than %d tracks" % min_next)

    raw_tracks = {}

    for pidx, playlist_file in enumerate(playlist_files):
        print("Processing ", playlist_file)
        with open(playlist_file, "r") as file:
            data = json.load(file)
            playlists = data["playlists"]
            jsonl_name = os.path.join(_OUTPUT_PATH.value, "%05d.jsonl" % pidx)
            with open(jsonl_name, "w") as file_writer:
                for playlist in playlists:
                    if playlist["num_tracks"] < min_next:
                        continue
                    tracks = playlist["tracks"]
                    track_context = []
                    artist_context = []
                    album_context = []
                    next_track = []
                    next_artist = []
                    next_album = []
                    for tidx, track in enumerate(tracks):
                        track_uri_idx = track_uri_dict[track["track_uri"]]
                        artist_uri_idx = artist_uri_dict[track["artist_uri"]]
                        album_uri_idx = album_uri_dict[track["album_uri"]]
                        if track_uri_idx not in raw_tracks:
                            raw_tracks[track_uri_idx] = track
                        if tidx < topk:
                            track_context.append(track_uri_idx)
                            artist_context.append(artist_uri_idx)
                            album_context.append(album_uri_idx)
                        else:
                            next_track.append(track_uri_idx)
                            next_artist.append(artist_uri_idx)
                            next_album.append(album_uri_idx)
                    assert len(next_track) > 0
                    assert len(next_artist) > 0
                    assert len(next_album) > 0
                    record = {
                        "track_context": track_context,
                        "album_context": album_context,
                        "artist_context": artist_context,
                        "next_track": next_track,
                        "next_album": next_album,
                        "next_artist": next_artist,
                    }
                    file_writer.write(json.dumps(record) + "\n")

    filename = os.path.join(_OUTPUT_PATH.value, "all_tracks.json")
    with open(filename, "w") as f:
        json.dump(raw_tracks, f)


if __name__ == "__main__":
    app.run(main)
