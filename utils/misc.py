import glob
import os
import pickle
import shutil
from subprocess import run

import numpy as np


def get_end(result_path):
    return len(list(result_path.joinpath("images").glob("*.png")))


def extract_frames(*, video_path, output_folder):
    return run(
        ["ffmpeg", "-i", video_path, "-vf","fps=30",output_folder.joinpath("%03d.png"), "-nostdin", "-loglevel", "error"],
        check=True,
    )


def copy_frames(*, image_dir_path, output_folder):
    i = 1
    print(image_dir_path)
    for pngfile in sorted(glob.iglob(os.path.join(str(image_dir_path), "*.png"))):
        shutil.copy(pngfile, output_folder.joinpath("{:03}.png".format(i)))
        i += 1

    return


# def create_video(*, images_folder, output_path):
#     return run(
#         ["ffmpeg", "-r", "60", "-pattern_type", "glob", "-i", images_folder, "-y", output_path, "-nostdin"], check=True
#     )

def create_video(*, images_folder, output_path):
    return run(
        # ["ffmpeg", "-framerate", "30","-f","image2", "-start_number", "1","-i", images_folder.joinpath("%03d.png"), "-y", output_path, "-nostdin", "-vcodec", "libx264",
        ["ffmpeg", "-framerate", "30", "-s", "700x466","-f","image2",  "-start_number", "30","-i", images_folder.joinpath("%03d.png"), "-y", output_path, "-nostdin", "-vcodec", "libx264",
        "-crf", "1" ,"-pix_fmt", "yuv420p"], check=True
    )



def compute_betas(*, rps_folder, beta_path):
    betas = []
    for hand in ["left", "right"]:
        pkl_files = list(rps_folder.joinpath(hand, "results").glob("*.pkl"))
        for pkl_file in pkl_files:
            with pkl_file.open("rb") as file:
                betas.append(pickle.load(file)["betas"][0])

    with beta_path.open("wb") as file:
        pickle.dump(np.median(betas, axis=0), file, pickle.HIGHEST_PROTOCOL)

def compute_expose_betas(*, expose_folder, beta_path):
    betas = []
    # for hand in ["left", "right"]:
    pkl_files = list(expose_folder.glob("*.pkl"))
    for pkl_file in pkl_files:
        with pkl_file.open("rb") as file:
            # breakpoint()
            betas.append(pickle.load(file)["betas"])

    with beta_path.open("wb") as file:
        pickle.dump(np.median(betas, axis=0), file, pickle.HIGHEST_PROTOCOL)
