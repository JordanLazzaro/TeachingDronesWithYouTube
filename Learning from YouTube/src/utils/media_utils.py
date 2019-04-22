"""
This is a set of utility functions for downloading and manipulating video/audio
"""
from pytube import YouTube
import skvideo.io  #requires 'brew install ffmpeg'
import torch
import librosa
import numpy as np
import os
from progressbar import ProgressBar
import json

def download_videos(URLs, DEST_PATH):
    """
    This function downloads the video given by URLs passed from YouTube.

    Params:
        -URLs: list of URLs to download
        -DEST_PATH: destination path for downloaded videos

    Returns:
        -Nothing
    """
    pbar = ProgressBar() # progress bar for downloading videos

    for url in pbar(URLs):
        try:
            yt = YouTube(url)
        except:
            print("Could not connect to YouTube...")

        # filter files for 'mp4' file type and get first file
        new_vid = yt.streams.filter(subtype='mp4').first()
        try:
            new_vid.download(DEST_PATH)
        except:
            print("Error downloading video...")

def video_to_array(VIDEO_FILE):
    """
    This function converts a mp4 file into a numpy array (takes a while)

    Params:
        -mp4 file

    Returns:
        -numpy array
    """
    return skvideo.io.vread(VIDEO_FILE)

def video_to_wave(VIDEO_FILE, timesteps):
    """
    This function returns a numpy array of sound bytes from a video
    """




def get_metadata(VIDEO_FILE):
    """
    This function returns a dict of the metadata for the mp4 file passed in.

    Params:
        -VIDEO_FILE: path to mp4 video

    Returns:
        -dict of metadata from file
    """
    return skvideo.io.ffprobe(video_location)

def print_metadata(metadata, mode='video'):
    """
    This function takes a dict of metadata and prints it to the terminal.

    Params:
        -metadata: dict of metadata to be printed
        -mode: type of metadata (either video or audio)

    Returns:
        -Nothing
    """
    if mode == 'video' or mode == 'audio':
        print("\n" + "---------- " + mode + " metadata ----------" + "\n")
        print(json.dumps(metadata[mode], indent=4))
        print("\n" + "---------- " + mode + " metadata ----------" + "\n")
    else:
        print("please enter valid mode (\"audio\" or \"video\")")

if __name__ == "__main__":

    destination = os.path.dirname(os.path.abspath(__file__))

    videos = ["https://www.youtube.com/watch?v=ToSe_CUG0F4"]

    # uncomment for downloading video
    # download_videos(videos, destination)

    # converts mp4 file to numpy array
    video_location = '/Users/jordanlazzaro/Research/projects/implementations/Learning from YouTube/src/utils/data/MONTEZUMAS REVENGE (ATARI 800XL).mp4'
    print_metadata(get_metadata(video_location), "audio")

    #video_array = video_to_array(video_location)
    #sound_bytes = get_sound_bytes(video_location)


    #print(type(video_array))
    #print(video_array.shape)
