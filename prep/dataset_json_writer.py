import os
import re
import json
import argparse

def count_sequential_false_images(root_dir):
    false_counts = {}

    for root, dirs, files in os.walk(root_dir):
        print(f"Checking {root, dirs}...")
        # sequential_dirs = [d for d in dirs if d.isdigit() and not re.match(r'0\d+', d)]
        sequential_dirs = [d for d in dirs if re.match(r'^[0-9]\d*$', d)]
        sequential_dirs.sort(key=int)  # Sort the sequential directories in ascending order
        for directory in sequential_dirs:
            subdir_path = os.path.join(root, directory)
            false_dir = os.path.join(subdir_path, "false")
            print(f"Checking {subdir_path}...")

            if os.path.isdir(false_dir):
                false_images = [file for file in os.listdir(false_dir) if file.lower().endswith(".jpg")]
                false_counts[subdir_path] = len(false_images)

    return false_counts

def check_all_false(root_dir):
    false_counts = {}

    for root, dirs, files in os.walk(root_dir):
        if "all_false" in root:
            false_images = [file for file in files if file.lower().endswith(".jpg")]
            false_counts[root] = len(false_images)
            print(f"Found {len(false_images)} false images in {root}")

    return false_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, default="dataset", help="Root directory of the dataset")
    args = parser.parse_args()
    root_directory = args.directory

    false_counts = check_all_false(root_directory)
    
    re_arg = fr"^{root_directory}/[^/]"
    false_counts = {dir: count for dir, count in false_counts.items() if re.match(re_arg, dir)
                    and "remove" not in dir}

    output_file = f"{root_directory}/dataset_false.json"  
    with open(output_file, "w") as f:
        json.dump(false_counts, f, indent=4)

    # false_counts = count_sequential_false_images(root_directory)
    # print(f"Total number of false images: {sum(false_counts.values())}")

    # re_arg = fr"^{root_directory}/[^/]+/\d+$"
    # # false_counts = {dir: count for dir, count in false_counts.items() if re.match(r'dataset/[^/]+/\d+$', dir)
    # #                 and "manual" not in dir}
    # false_counts = {dir: count for dir, count in false_counts.items() if re.match(re_arg, dir)
    #                 and "remove" not in dir}

    # for subdir, count in false_counts.items():
    #     print(f"Directory '{subdir}' has {count} false images.")

    # output_file = f"{root_directory}/dataset_false.json"

    # with open(output_file, "w") as f:
    #     json.dump(false_counts, f, indent=4)

    # print(f"Data written to {output_file}")
