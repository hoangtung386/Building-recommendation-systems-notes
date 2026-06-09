#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Given Json playlist files makes dictionaries of items.
"""

import glob
import json
import os
from typing import Any, Dict, Tuple

from absl import app
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS
_PLAYLISTS = flags.DEFINE_string("playlists", None, "Playlist json glob.")
_OUTPUT_PATH = flags.DEFINE_string("output", "data", "Output path.")

flags.mark_flag_as_required("playlists")


def update_dict(dict: Dict[Any, int], item: Any):
    if item not in dict:
        index = len(dict)
        dict[item] = index


def dump_dict(dict: Dict[str, str], name: str):
    fname = os.path.join(_OUTPUT_PATH.value, name)
    with open(fname, "w") as f:
        json.dump(dict, f)


def main(argv):
    """Main function."""
    del argv

    playlist_files = glob.glob(_PLAYLISTS.value)
    track_uri_dict = {}
    artist_uri_dict = {}
    album_uri_dict = {}

    for playlist_file in playlist_files:
        print("Processing ", playlist_file)
        with open(playlist_file, "r") as file:
            data = json.load(file)
            playlists = data["playlists"]
            for playlist in playlists:
                tracks = playlist["tracks"]
                for track in tracks:
                    update_dict(track_uri_dict, track["track_uri"])
                    update_dict(artist_uri_dict, track["artist_uri"])
                    update_dict(album_uri_dict, track["album_uri"])

    dump_dict(track_uri_dict, "track_uri_dict.json")
    dump_dict(artist_uri_dict, "artist_uri_dict.json")
    dump_dict(album_uri_dict, "album_uri_dict.json")


if __name__ == "__main__":
    app.run(main)
