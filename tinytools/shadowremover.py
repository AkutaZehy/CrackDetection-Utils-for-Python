import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import os

class GrayShadowProcess:
    @staticmethod
    def mmClose(img, radius=3):
        """
        Performs morphological closing operation on the input image.

        Args:
            img (numpy.ndarray): Input grayscale image.
            radius (int, optional): Radius of the structuring element. Defaults to 3.

        Returns:
            numpy.ndarray: Closed image.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return closed

    @staticmethod
    def gauSmooth(img, radius=30, sigma=3):
        """
        Performs Gaussian smoothing on the input image.

        Args:
            img (numpy.ndarray): Input image.
            radius (int, optional): Radius of the Gaussian kernel. Defaults to 30.
            sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 3.

        Returns:
            numpy.ndarray: Smoothed image.
        """
        smoothed = gaussian_filter(img, sigma=sigma, radius=radius)
        return smoothed

    @staticmethod
    def geoLevel(img, N=10, L=7):
        """
        Divides the input image into N levels and returns the first L levels and the remaining levels.

        Args:
            img (numpy.ndarray): Input image.
            N (int, optional): Number of levels to divide the image into. Defaults to 10.
            L (int, optional): Number of levels to keep. Defaults to 7.

        Returns:
            tuple: A tuple containing two numpy arrays - the first L levels and the remaining levels.
        """
        ng = img.size // N
        levels = []
        i = 0
        sum = 0
        for k in range(256):
            Pk = (img == k)
            if sum == 0:
                levels.append(Pk)
            else:
                levels[i] = np.logical_or(levels[i], Pk)
            sum += np.sum(Pk)
            if sum >= ng:
                i += 1
                sum = 0
            if i >= N:
                break
        S = levels[:L]
        B = np.logical_or.reduce(levels[L:])
        return S, B

    @staticmethod
    def illumCompensate(img, S, B):
        """
        Performs illumination compensation on the input image.

        Args:
            img (numpy.ndarray): Input image.
            S (numpy.ndarray): First L levels of the image.
            B (numpy.ndarray): Remaining levels of the image.

        Returns:
            numpy.ndarray: Compensated image.
        """
        DB = np.std(img[B])
        IB = np.mean(img[B])
        for level in S:
            mask_S = level.astype(bool)
            DS = np.std(img[mask_S])
            IS = np.mean(img[mask_S])
            alpha = DB / DS if DS != 0 else 0
            lambda_ = IB - alpha * IS
            img[mask_S] = alpha * img[mask_S] + lambda_
        return img

    @staticmethod
    def process_image(image_path, radius=3, smooth_radius=30, smooth_sigma=3, N=10, L=7):
        """
        Process the input image by performing morphological closing, Gaussian smoothing,
        dividing into levels, and illumination compensation.

        Args:
            image_path (str): Path to the input image.
            radius (int, optional): Radius of the structuring element for morphological closing. Defaults to 3.
            smooth_radius (int, optional): Radius of the Gaussian kernel for smoothing. Defaults to 30.
            smooth_sigma (float, optional): Standard deviation of the Gaussian kernel for smoothing. Defaults to 3.
            N (int, optional): Number of levels to divide the image into. Defaults to 10.
            L (int, optional): Number of levels to keep. Defaults to 7.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        closed = GrayShadowProcess.mmClose(img, radius=radius)
        smoothed = GrayShadowProcess.gauSmooth(closed, radius=smooth_radius, sigma=smooth_sigma)
        S, B = GrayShadowProcess.geoLevel(smoothed, N=N, L=L)
        compensated = GrayShadowProcess.illumCompensate(img, S, B)
        
        return compensated
    
def batch_process_images(input_path, output_path, radius=3, smooth_radius=30, smooth_sigma=3, N=10, L=7):
        # Check if output path exists
        if os.path.exists(output_path):
            print("Warning: Output path already exists. Deleting existing files.")
            # Delete existing files in output path
            file_list = os.listdir(output_path)
            for file_name in file_list:
                file_path = os.path.join(output_path, file_name)
                os.remove(file_path)
        else:
            # Create output path
            os.makedirs(output_path)

        # Get list of files in input path
        file_list = os.listdir(input_path)

        # Process each image file
        for file_name in file_list:
            file_path = os.path.join(input_path, file_name)
            if os.path.isfile(file_path):
                # Check if file is an image
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # Process image
                    compensated = GrayShadowProcess.process_image(file_path, radius=radius, smooth_radius=smooth_radius, smooth_sigma=smooth_sigma, N=N, L=L)

                    # Save processed image to output path with same name
                    output_file_path = os.path.join(output_path, file_name)
                    cv2.imwrite(output_file_path, compensated)
                else:
                    print(f"Skipping file: {file_name} (not an image)")

        print("Batch processing completed.")

def main():
    # Usage example
    input_path = 'input'
    output_path = 'output'
    batch_process_images(input_path, output_path)

if __name__ == "__main__":
    main()
