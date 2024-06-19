import numpy as np
import matplotlib.pyplot as plt
import glob

MODEL = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/StarDist3D_complete_1-1_42"

def video_loop(model=MODEL, num_videos=5):
    videos = glob.glob(f"{model}/*.npz")
    videos = np.random.choice(videos, size=num_videos)
    temporary = np.zeros((num_videos, 600, 512, 512))
    for i, fvid in enumerate(videos):
        vid = np.load(fvid)["label"]
        temporary[i] = vid
    print(temporary)

def main():
    video_loop()

if __name__=="__main__":
    main()