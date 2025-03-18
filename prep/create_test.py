import os
import glob
import argparse
import random
import shutil

def create_max_dir(args):
    max_dir = 0
    for root, dirs, files in os.walk(f"{args.directory}/{args.game}"):
        sequential_dirs = [d for d in dirs if d.isdigit()]
        max_dir = max(max_dir, max([int(d) for d in sequential_dirs], default=0))
        for directory in sequential_dirs:
            try:
                os.makedirs(f"{args.directory}/{args.game}/test/{directory}/true", exist_ok=True)
            except FileExistsError:
                pass
            try:
                os.mkdir(f"{args.directory}/{args.game}/test/{directory}/false")
            except FileExistsError:
                pass
        if not sequential_dirs:
            continue
    return max_dir

def get_false_images(directory, count=20):
    false_images = glob.glob(f"{directory}/*.jpg")
    # randomly get 20 false images quickly
    random.shuffle(false_images)
    false_images = false_images[:count]
    return false_images

def main(args):
    max_dir = create_max_dir(args)

    for i in range(0, max_dir+1):
        print(f"Creating test dataset for {args.game} in {args.directory}/{args.game}/test/{i}...")
        false_images = []
        true_images = []
        for root, dirs, files in os.walk(f"{args.directory}"):
            if "all_false" in root and args.game not in root:
                false_img = get_false_images(root, count=20)
                false_images.extend(false_img)
        false_img = get_false_images(f"{args.directory}/{args.game}/{i}/false", count=40)
        false_images.extend(false_img)
        true_img = get_false_images(f"{args.directory}/{args.game}/{i}/true", count=len(false_images))

        for img in false_images:
            shutil.copy(img, f"{args.directory}/{args.game}/test/{i}/false")
        for img in true_img:
            shutil.copy(img, f"{args.directory}/{args.game}/test/{i}/true")
        
        false_images = glob.glob(f"{args.directory}/{args.game}/test/{i}/false/*.jpg")
        true_images = glob.glob(f"{args.directory}/{args.game}/test/{i}/true/*.jpg")

        print(f"Total false images: {len(false_images)}\n"
              f"Total true images: {len(true_images)}")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, default="dataset", help="Root directory of the dataset")
    parser.add_argument("-g", "--game", type=str, required=True, help="Game name")
    args = parser.parse_args()

    main(args)