import re
import subprocess

def parse_v4l2_output(device, control_name):

    output = get_v4l2(device)
    pattern = rf"{control_name}\s+0x[0-9a-f]+.*?min=(-?\d+)\s+max=(-?\d+)"
    match = re.search(pattern, output, re.IGNORECASE)
    if match:
        min_val, max_val = map(int, match.groups())
        return min_val, max_val
    else:
        raise ValueError(f"Control '{control_name}' not found.")

def get_v4l2(device):
    script = f"v4l2-ctl --list-ctrls -d {device}"



    result = subprocess.run(script, shell=True, capture_output=True, text=True)
    return result.stdout

if __name__ == "__main__":

    brightness_min_max = parse_v4l2_output('0', "brightness")
    contrast_min_max = parse_v4l2_output('0', "contrast")

    print(f"Brightness: min={brightness_min_max[0]}, max={brightness_min_max[1]}")
    print(f"Contrast: min={contrast_min_max[0]}, max={contrast_min_max[1]}")
