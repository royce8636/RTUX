import os
import json

def main(root_dir):
    if not root_dir.endswith('/'):
        root_dir += '/'
    data = {}

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if '_threshold' in file:
                file_path = os.path.join(subdir, file)
                print(file_path)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # remove empty lines
                    lines = [line for line in lines if line.strip()]
                    lines = [float(line.strip()) for line in lines]
                    for i, value in enumerate(lines):
                        key = file.replace('threshold', f"{i}")
                        key = key.replace('.txt', '')
                        data[key] = value

    with open(f'{root_dir}thresholds.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(data)

    return data

if __name__ == '__main__':
    main('dataset/')
