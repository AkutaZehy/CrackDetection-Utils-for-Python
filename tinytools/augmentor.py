import shutil
import cv2
import os
import random

import numpy as np

class Augmentor:
    def __init__(self, path, aug_path):
        self.path = path
        self.aug_path = aug_path
        self.images_path = os.path.join(path, "images")
        self.masks_path = os.path.join(path, "masks")
        self.aug_images_path = os.path.join(aug_path, "images")
        self.aug_masks_path = os.path.join(aug_path, "masks")
        self.augmented_count = 0

    def check_dir(self, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def count_cracks(self, image_path):
        # Read the image
        image = cv2.imread(image_path, 0)  # Read as grayscale

        # Convert the image to binary
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count the number of cracks
        num_cracks = len(contours)

        return num_cracks, contours

    def calculate_channel_median(self, image_path):
        # Read the image
        image = cv2.imread(image_path)

        # Split the image into RGB channels
        b, g, r = cv2.split(image)

        # Calculate the median value for each channel
        median_r = np.median(r)
        median_g = np.median(g)
        median_b = np.median(b)

        return median_r, median_g, median_b

    def augment_images(self):
        self.check_dir(self.aug_path)
        self.check_dir(self.aug_images_path)
        self.check_dir(self.aug_masks_path)

        for filename in os.listdir(self.images_path):
            # Check if there is a corresponding mask file
            mask_filename = os.path.join(self.masks_path, filename)
            if not os.path.isfile(mask_filename):
                continue

            # Check if the image is all black
            image = cv2.imread(os.path.join(self.images_path, filename))
            mask = cv2.imread(mask_filename, 0) 
            if cv2.countNonZero(mask) == 0:
                continue

            # Calculate the number of cracks
            num_cracks, contours = self.count_cracks(mask_filename)

            if num_cracks == 1:
                continue

            # Calculate the median value for each channel
            median = self.calculate_channel_median(os.path.join(self.images_path, filename))

            # Randomly select a subset of cracks to erase
            num_to_erase = int(num_cracks / 2)
            cracks_to_erase = random.sample(range(num_cracks), num_to_erase)

            # Load the mask image
            mask = cv2.imread(mask_filename, 0)

            # Erase the selected cracks in the mask
            for i in cracks_to_erase:
                background = np.zeros_like(image)
                background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
                cv2.drawContours(background, [contours[i]], -1, 255, thickness=cv2.FILLED)

                # Dilate the cracks in the background image
                background = cv2.dilate(background, np.ones((5, 5), np.uint8), iterations=1)
                _, background = cv2.threshold(background, 1, 255, cv2.THRESH_BINARY)

                # Fill the selected cracks in the image
                filled_image = image.copy()
                filled_image[background == 255] = median
                filled_image = cv2.inpaint(filled_image, background, 3, cv2.INPAINT_TELEA)

                # Replace the original image with the filled image
                image = filled_image
                cv2.drawContours(mask, [contours[i]], -1, 0, thickness=cv2.FILLED)

            # Save the modified image and mask in the augmented directory
            cv2.imwrite(os.path.join(self.aug_images_path, filename), image)
            cv2.imwrite(os.path.join(self.aug_masks_path, filename), mask)

            self.augmented_count += 1  # Increment the count of augmented images

        print("Total number of augmented images:", self.augmented_count)

def main():
    path = "input"  # 替换为你的路径
    aug_path = "augmented"

    augmentor = Augmentor(path, aug_path)
    augmentor.augment_images()

if __name__ == "__main__":
    main()
