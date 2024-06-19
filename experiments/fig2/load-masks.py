
import numpy
import tifffile
import os
import argparse

PATH = "./data"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-id", type=str, help="Movie ID")
    args = parser.parse_args()

    mask = numpy.load(os.path.join(PATH, f"mask-{args.movie_id}.npz"))["label"] > 0
    tifffile.imwrite(os.path.join(PATH, f"mask-{args.movie_id}.tif"), mask.astype(numpy.uint8))


if __name__ == "__main__":
    main()