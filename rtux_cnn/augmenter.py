import cv2
import argparse
import os
import numpy as np
import random
import itertools

try:
    from rtux_cnn.utils import print
except ImportError:
    from utils import print

shape = 128


class Augmenter:
    def half_grad(self, img):
        start = random.choice([0, 1])
        end = 0 if start == 1 else 1
        range = random.randint(56, 100)
        exponential_values = np.linspace(start, end, range) ** 3
        # if start == 1:
        #     mask = np.repeat(np.tile(np.concatenate((np.ones(shape - range), exponential_values)), (shape, 1))[:, :,
        #                      np.newaxis], 3,
        #                      axis=2)
        # else:
        #     mask = np.repeat(np.tile(np.concatenate((exponential_values, np.ones(shape - range))), (shape, 1))[:, :,
        #                      np.newaxis], 3,
        #                      axis=2)
                
        if start == 1:
            mask = np.tile(np.concatenate((np.ones(shape - range), exponential_values)), (shape, 1))
        else:
            mask = np.tile(np.concatenate((exponential_values, np.ones(shape - range))), (shape, 1))

        # print(mask.shape)
        adjusted_image = np.uint(img * mask)
        return adjusted_image


    # def half_grad(self, img):
    #     start = random.choice([0, 1])
    #     end = 0 if start == 1 else 1
    #     height, width = img.shape
    #     channels = 1
    #     range_size = random.randint(56, 100)
    #     exponential_values = np.linspace(start, end, range_size) ** 3
        
    #     if start == 1:
    #         mask = np.tile(np.concatenate((np.ones(width - range_size), exponential_values)), (height, 1))
    #     else:
    #         mask = np.tile(np.concatenate((exponential_values, np.ones(width - range_size))), (height, 1))
        
    #     # Repeat the mask for each channel
    #     mask = np.stack([mask] * channels, axis=2)

    #     adjusted_image = np.uint8(img * mask)
    #     return adjusted_image


    def create_aug(self, img, grad=False):
        # img = cv2.resize(img, (shape, shape))
        img = img.numpy()  # Convert TensorFlow tensor to numpy array

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # if np.mean(img) > 100:
        #     brightness = random.randint(-8, -1)
        #     contrast = round(random.uniform(0.9, 1.2), 1)
        #     adjusted_image = cv2.addWeighted(img, contrast, img, 0, brightness)
        if np.mean(img) > 150:
            brightness = random.randint(-6, -1)
            contrast = round(random.uniform(0.95, 1.1), 1)
            adjusted_image = cv2.addWeighted(img, contrast, img, 0, brightness)
        else:
            brightness = random.randint(10, 18)
            contrast = round(random.uniform(0.9, 1.2), 1)
            adjusted_image = cv2.addWeighted(img, contrast, img, 0, brightness)
        shift_x = random.randint(1, int(shape * 0.095)) * random.choice([-1, 1])
        shift_y = random.randint(1, int(shape * 0.095)) * random.choice([-1, 1])
        gray_intensity = np.random.randint(0, 256)
        fill_gray = (gray_intensity, gray_intensity, gray_intensity)

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        adjusted_image = cv2.warpAffine(adjusted_image, M,
                                        (adjusted_image.shape[1], adjusted_image.shape[0]),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=fill_gray)
        if grad is True:
            adjusted_image = self.half_grad(adjusted_image)
        # make it float32
        adjusted_image = adjusted_image.astype('float32')
        # normalize between 0 and 1
        adjusted_image /= 255
        # print(adjusted_image.shape)
        return adjusted_image[..., np.newaxis]

    def augment_count(self, target_path, save_path, num, save_org=False):
        print(f"Augmenting {target_path} to {save_path}, {num} images")
        save_name = '_'.join(target_path.split("/")[1:])
        if not os.path.exists(target_path):
            print(f"Can't find {target_path}")
            return 1
        os.makedirs(save_path, exist_ok=True)
        cur_count = len(os.listdir(target_path))
        augment = True
        if cur_count >= num:
            print(f"Already have {cur_count} images, skipping")
            augment = False
        cycle_files = itertools.cycle(os.listdir(target_path))
        if save_org:
            for idx, i in enumerate(os.listdir(target_path)[:num]):
                img = cv2.imread(f"{target_path}/{i}")
                img = cv2.resize(img, (shape, shape))
                if (idx + 1) % 10 == 0 and "true" not in save_path:
                    img = self.half_grad(img)
                cv2.imwrite(f"{save_path}/{save_name}_org{idx}.jpg", img)
        if augment is True:
            for i in range(num - cur_count):
                filename = next(cycle_files)
                img = cv2.imread(f"{target_path}/{filename}")
                grad = True if i % 5 == 0 and "true" not in save_path else False
                cv2.imwrite(f"{save_path}/{save_name}_aug{i}.jpg", self.create_aug(img, grad))

    def check_dir(self, save_path):
        target = os.path.basename(save_path).split('-')[0]
        dep_dir = os.path.basename(save_path).split('-')[1].split(',')
        existing_dir = [i for i in os.listdir(os.path.dirname(save_path)) if
                        target in i]
        min_missing_count = float('inf')
        selected_existing_dir = None
        selected_missing = None

        for existing_dir in existing_dir:
            dependent_dirs = existing_dir.split('-')[1].split(',')
            missing_elements = [element for element in dep_dir if element not in dependent_dirs]
            if any(element not in dep_dir for element in dependent_dirs):
                continue  # Skip
            missing_count = len(missing_elements)
            print(dependent_dirs)
            print("Missing elements:", missing_elements)

            if missing_count < min_missing_count:
                min_missing_count = missing_count
                selected_existing_dir = existing_dir
                selected_missing = missing_elements
        if selected_existing_dir:
            print("Selected Existing Directory:", selected_existing_dir)
            return selected_existing_dir, selected_missing
        else:
            print("No existing directory with missing elements")
            return None, None

    def match_ratio(self, false_path: list, true_path: str, save_path: str, ratio: list, save_org=False, false_ratio:
    list = None, existing: dict = None):
        if false_ratio is not None and len(false_ratio) != len(false_path):
            print("False ratio must be same length as false path")
            return 1
        # Check path existence
        if not os.path.exists(true_path) or any(not os.path.exists(path) for path in false_path):
            print("One or more directories not found")
            print(f"Given paths:\n"
                  f"True: {true_path}\nFalse: {false_path}")
            return 1
        # existing_dir, missing = self.check_dir(save_path)
        # if existing_dir:

        # if os.path.exists(save_path):
        #     print(f"Save path {save_path} and its contents will all be deleted")
        #     os.system(f"rm -rf {save_path}")
        # Get current count
        true_count = len(os.listdir(true_path))
        false_count = sum(len(os.listdir(path)) for path in false_path)
        # Check if correct ratio is given
        if any(value < 0 for value in ratio):
            print("Can't have negative ratio")
            return 1
        # Adjust ratio to be integers
        if any(value % 1 != 0 for value in ratio):
            ratio = [round(i / min(ratio)) for i in ratio]
        cur_ratio = false_count / true_count
        des_ratio = ratio[0] / ratio[1]
        if cur_ratio > des_ratio:
            required_true = int(false_count / des_ratio)
            additional_true = max(0, required_true - true_count)
            additional_false = 0
        else:
            required_false = int(true_count * des_ratio)
            additional_false = max(0, required_false - false_count)
            additional_true = 0

        # Calculate how much additional images are needed for each dir in false_path
        final_false, final_true = additional_false + false_count, additional_true + true_count

        if final_false + final_true < 2000:
            final_true = int(2000 * (final_true / (final_false + final_true)))
            final_false = 2000 - final_true
        if false_ratio is None:
            final_false = [int(final_false * (len(os.listdir(path)) / false_count)) for path in false_path]
        elif false_ratio is not None:
            false_rat_sum = sum(false_ratio)
            final_false = [int(false_ratio[i] / false_rat_sum * final_false) for i in range(len(false_ratio))]

        # Exclude already trained images from count
        final_false = [count - existing[path] if existing is not None and path in existing else count for path, count in
                       zip(false_path, final_false)]
        final_true = final_true - existing[true_path] if existing is not None and true_path in existing else final_true

        # Create save_path if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        print(f"Final image counts: True: {final_true}, False: {final_false}")
        self.augment_count(true_path, f"{save_path}/true", final_true, save_org)
        [self.augment_count(path, f"{save_path}/false", count, save_org) for path, count in zip(false_path,
                                                                                                final_false)]
        print(
            f"Final directory count: True: {len(os.listdir(f'{save_path}/true'))}, False: {len(os.listdir(f'{save_path}/false'))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, required=True,
                        help='Directory of source photos')
    parser.add_argument('-s', '--save', type=str, required=True,
                        help='Where to save the photos')
    parser.add_argument('-r', '--resize', type=str, required=True,
                        help='Resize ratio (false:true)')
    parser.add_argument('-o', '--org', action='store_true',
                        help='Whether to save original images')
    args = vars(parser.parse_args())

    if ':' in args['resize']:
        args['resize'] = [int(i) for i in args['resize'].split(':')]
    else:
        print("Wrong resize format. Ratio should be in the form of 'false:true'")
        exit(1)

    aug = Augmenter()
    # aug.check_dir("dataset/supercell/supercell_0-false,black,white,792426e2,supercell_1", )
    # aug.match_ratio(['dataset/black', 'dataset/HOMESCREEN', 'dataset/supercell/0/false'], 'dataset/supercell/0/true',
    #                 args['save'], args['resize'], args['org'], [1, 1, 3], {'dataset/supercell/0/false': 45,
    #                                                                        'dataset/supercell/0/true': 214})
    # aug.create_aug()
    aug.augment_count(args['directory'], args['save'], 2000, args['org'])
