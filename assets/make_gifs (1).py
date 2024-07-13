import os
import numpy as np
import cv2
from PIL import Image, ImageSequence
import imageio


def create_gifs(video_folder):
    for fn in os.listdir(video_folder):
        output_gif_fn = fn[:-4] + '.gif'
        cmd = 'ffmpeg -y -i %s -vf "fps=10,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 %s'%(os.path.join(video_folder, fn), os.path.join(video_folder, output_gif_fn))
        #cmd = 'ffmpeg -y -i %s -vf "crop=1200:1080:360:0,eq=gamma=1.4:saturation=1.3,fps=10,scale=400:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 %s'%(os.path.join(video_folder, fn), os.path.join(video_folder, output_gif_fn)) # for cropping / brightness increase for dessert plates
        os.system(cmd)

if __name__ == '__main__':
    create_gifs('spaghetti/edits')

