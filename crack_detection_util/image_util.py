import math
import os
import random
import shutil
import cv2
import datetime
import json
import time

import numpy as np

class MyImage:
    def __init__(self, folder, model_name='unet', chunk_size=512, input_path=None, output_path=None, cache_path=None):
        """
        Initialize MyImage object.

        Args:
            folder (str): The path to the folder. If provided, input_path, output_path, and cache_path will be set as subfolders of the folder. Or set this parameter to None and provide input_path, output_path, and cache_path instead.
            model_name (str, optional): The name of the model to use for prediction. Defaults to 'unet'.
            chunk_size (int, optional): The size of each chunk. Defaults to 512.
            input_path (str optional): The path to the input folder. It is not allowed to skip this parameter if folder is None.
            output_path (str, optional): The path to the output folder. If not provided, it will be set to the same as the input path.
            cache_path (str, optional): The path to the cache folder. If not provided, it will be set to the same as the output path.
        """
        if folder is not None:
            self.use_folder_strategy = True
            self.__input = os.path.join(folder, 'input')
            self.__output = os.path.join(folder, 'output')
            self.__cache = os.path.join(folder, 'cache')
        else:
            self.use_folder_strategy = False
            if input_path is None:
                raise ValueError("input_path is required if folder is None.")
            self.__input = input_path
            self.__output = output_path if output_path else input_path
            self.__cache = cache_path if cache_path else self.__output
        self.__chunk_size = chunk_size
        self.model_name = model_name.lower()

        self.check(output_info=False)
    
    def __str__(self):
        """
        Return a string representation of the MyImage object.

        Returns:
            str: The string representation of the MyImage object.
        """
        return f"MyImage Object:\n" \
               f"Folder Strategy: {self.use_folder_strategy}\n" \
               f"Input Path: {self.__input}\n" \
               f"Output Path: {self.__output}\n" \
               f"Cache Path: {self.__cache}\n" \
               f"Chunk Size: {self.__chunk_size}\n" \
               f"Model Name: {self.model_name}\n"
    
    def check(self, create_folders=True, output_info=True):
        """
        Check if the input, output, and cache folders exist. Create them if necessary.

        Args:
            create_folders (bool, optional): Whether to create the folders if they don't exist. Defaults to True.
            output_info (bool, optional): Whether to output information about the folders. Defaults to True.
        """
        folders = [self.__input, self.__output, self.__cache]

        for folder in folders:
            if not os.path.exists(folder):
                if create_folders:
                    os.makedirs(folder)
                    if output_info:
                        print(f"Created folder: {os.path.abspath(folder)}")
                else:
                    raise Exception(f"Folder does not exist! Path: {os.path.abspath(folder)}")
            else:
                if output_info:
                    print(f"Folder already exists: {os.path.abspath(folder)}")

    def clear_cache(self, clear_output=True):
        """
        Clear the cache folder. If the cache folder does not exist, create it. If it is already cleared, only output the message.

        Args:
            clear_output (bool, optional): Whether to clear the output folder after clearing the cache. Defaults to True.
        """
        if os.path.exists(self.__cache):
            shutil.rmtree(self.__cache)
            os.makedirs(self.__cache)

        if clear_output: print("Cache folder clear.")

    def convert(self, format='jpg'):
        """
        Convert images in the input folder to the specified format and save them in the output/timestamp-convert folder.

        Args:
            format (str, optional): The format to convert the images to. Defaults to 'jpg'.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S%f")
        convert_folder = os.path.join(self.__output, f"{timestamp}-convert")
        os.makedirs(convert_folder, exist_ok=True)

        for file in os.listdir(self.__input):
            file_path = os.path.join(self.__input, file)
            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                converted_file_path = os.path.join(convert_folder, f"{os.path.splitext(file)[0]}.{format}")
                cv2.imwrite(converted_file_path, image)

        print(f"Images converted and saved in: {os.path.abspath(convert_folder)}")

    def batch_chunk_images(self, input_path=None, output_path=None, chunk_size=None, overwrite=True):
        """
        Batch chunk images by iterating through the input folder and calling the chunk method from ChunkAndMerge.

        Args:
            input_path (str, optional): The path to the input folder. Defaults to 'input'.
            output_path (str, optional): The path to the output folder. Defaults to 'cache/chunk'.
            chunk_size (int, optional): The size of each chunk. Defaults to 512.
            clear_output_folder (bool, optional): Whether to clear the output folder before chunking. Defaults to True.
        """
        self.check(create_folders=False, output_info=False)
        input_path = input_path if input_path else self.__input
        output_path = output_path if output_path else os.path.join(self.__cache, 'chunk')
        chunk_size = chunk_size if chunk_size else self.__chunk_size
        for file in os.listdir(input_path):
            file_path = os.path.join(input_path, file)
            chunk_and_merge = ChunkAndMerge(file_path, output_path, chunk_size=chunk_size)
            if os.path.isfile(file_path):
                chunk_and_merge.chunk(overwrite=overwrite)
        
        print(f"Images chunked and saved in: {os.path.abspath(output_path)}")

    def batch_merge_images(self, input_path=None, output_path=None, overwrite=True):
        """
        Batch merge the chunked images in the input folder and save the merged images in the output folder.

        Args:
            input_path (str, optional): The path to the input folder. Defaults to 'cache/prediction'.
            output_path (str, optional): The path to the output folder. Defaults to 'output/prediction_unet'.
        """
        input_path = input_path if input_path else os.path.join(self.__cache, 'prediction')
        output_path = output_path if output_path else os.path.join(self.__output, 'prediction_' + self.model_name)
        self.check(create_folders=False, output_info=False)
        chunk_and_merge = ChunkAndMerge(input_path, output_path)
        for folder in os.listdir(input_path):
            input_folder_path = os.path.join(input_path, folder)
            output_folder_path = output_path
            os.makedirs(output_folder_path, exist_ok=True)
            chunk_and_merge.merge(input_folder_path, output_folder_path, overwrite=overwrite)

        print(f"Images merged and saved in: {os.path.abspath(output_path)}")

    def batch_predict_images(self, predict, input_path=None, output_path=None, size=None):
        """
        Batch predict the images in the input folder using the provided predict function and save the predicted images in the output folder.

        Args:
            predict (function): The predict function to use for image prediction. It should take the input folder, output folder, and size as parameters.
            input_path (str, optional): The path to the input folder. Defaults to 'cache/chunk'.
            output_path (str, optional): The path to the output folder. Defaults to 'cache/prediction'.
            size (int, optional): The image size used for predicting model training. Defaults to chunk_size(default is 512), which means the default size of a picture is 512x512.
        """
        input_path = input_path if input_path else os.path.join(self.__cache, 'chunk')
        output_path = output_path if output_path else os.path.join(self.__cache, 'prediction')
        size = size if size else self.__chunk_size
        self.check(create_folders=False, output_info=False)

        for folder in os.listdir(input_path):
            folder_path = os.path.join(input_path, folder)
            if os.path.isdir(folder_path):
                output_folder_path = os.path.join(output_path, folder)
                os.makedirs(output_folder_path, exist_ok=True)
                info_file_path = os.path.join(folder_path, 'info.json')
                if os.path.isfile(info_file_path):
                    shutil.copy(info_file_path, output_folder_path)
                else:
                    raise FileNotFoundError(f"info.json file not found in folder: {folder_path}")
                for image in os.listdir(folder_path):
                    if os.path.basename(image) == "info.json": continue
                    input_image_path = os.path.join(folder_path, image)
                    if os.path.isfile(input_image_path):
                        predict(input_image_path, output_folder_path, size)

        print(f"Images predicted and saved in: {os.path.abspath(output_path)}")

    def split_file(self, input_path=None, rate=0.2):
        """
        Split the input folder into training and validation sets and save them in the output folder.

        Args:
            input_path (str, optional): The path to the input folder. Defaults to 'input'.
            rate (float, optional): The rate of the validation set. Defaults to 0.2. If the rate is greater than 1, it will be treated as the size of the validation set.
        """

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S%f")
        
        # Check if the input folder exists
        input_path = input_path if input_path else self.__input

        img_path = [os.path.join(input_path, img) for img in os.listdir(input_path) if img.endswith('.jpg')]
        random.shuffle(img_path)

        # Split the dataset
        val_size = int(len(img_path) * (1 - rate)) if rate < 1 else rate
        if val_size is not int or rate < 0:
            raise ValueError("Invalid rate value. Please use a positive integer or a float between 0 and 1.")
        val_img_path = img_path[:val_size]
        train_img_path = img_path[val_size:]

        output_path = os.path.join(self.__output, f"{timestamp}-split")
        val_path = output_path + '/val/'
        train_path = output_path + '/train/'
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(train_path, exist_ok=True)

        for img in val_img_path:
            shutil.copy(img, val_path)

        for img in train_img_path:
            shutil.copy(img, train_path)

class ChunkAndMerge:
    def __init__(self, input_path, output_path, print_info=False, chunk_size=512):
        """
        Initialize ChunkAndMerge object.

        Args:
            input_path (str): The path to the input file or folder.
            output_path (str): The path to the output file or folder.
            print_info (bool, optional): Whether to print information about the folders. Defaults to False.
        """
        self.__input = input_path
        self.__output = output_path
        self.__print_info = print_info
        self.__chunk_size = chunk_size
        self.check()

    def check(self, create_folders=True):
        """
        Check if the input and output folders exist. Create them if necessary.

        Args:
            create_folders (bool, optional): Whether to create the folders if they don't exist. Defaults to True.
            output_info (bool, optional): Whether to output information about the folders. Defaults to True.
        """
        folders = [self.__input, self.__output]

        for folder in folders:
            if not os.path.exists(folder):
                if create_folders:
                    os.makedirs(folder)
                    if self.__print_info:
                        print(f"Created folder: {os.path.abspath(folder)}")
                else:
                    raise Exception(f"Folder does not exist! Path: {os.path.abspath(folder)}")
            else:
                if self.__print_info:
                    print(f"Folder already exists: {os.path.abspath(folder)}")

    def switch_print_info(self):
        """
        Switch the print_info attribute between True and False.
        """
        self.__print_info = not self.__print_info

    def chunk(self, chunk_size=None, output_exceptions=True, overwrite=True, warn_existing_files=True):
        """
        Chunk the input image into multiple images in the output folder.

        Args:
            chunk_size (int, optional): The size of each chunk. Defaults to 512.
            skip_exceptions (bool, optional): Whether to skip exceptions when encountering problematic images. Defaults to True.
            output_exceptions (bool, optional): Whether to output exceptions when encountering problematic images. Defaults to True.
            overwrite (bool, optional): Whether to clear the output folder before saving the chunked images. Defaults to True.
            warn_existing_files (bool, optional): Whether to output a warning when there are existing files in the output folder. Defaults to True.
        """
        chunk_size = chunk_size if chunk_size else self.__chunk_size

        if overwrite:
            for file in os.listdir(self.__output):
                file_path = os.path.join(self.__output, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        file_path = self.__input
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            if image is None:
                if output_exceptions:
                    print(f"Skipping invalid image: {file_path}")
                return

            # Create the output folder
            output_folder = os.path.join(self.__output, os.path.splitext(os.path.basename(file_path))[0])
            os.makedirs(output_folder, exist_ok=True)

            if warn_existing_files and len(os.listdir(output_folder)) > 0:
                print(f"Warning: Existing files found in output folder: {output_folder}")

            # Generate info.json
            height, width, _ = image.shape
            info = {
                "width": width,
                "height": height,
                "chunk_size": self.__chunk_size,
                "original_name": os.path.basename(file_path),
                "original_format": os.path.splitext(file_path)[1][1:]
            }
            info_file_path = os.path.join(output_folder, "info.json")
            with open(info_file_path, "w") as info_file:
                json.dump(info, info_file)

            # Update the height and width
            if height % self.__chunk_size != 0 or width % self.__chunk_size != 0:
                # Calculate the number of pixels to add
                add_height = self.__chunk_size - (height % self.__chunk_size) if height % self.__chunk_size != 0 else 0
                add_width = self.__chunk_size - (width % self.__chunk_size) if width % self.__chunk_size != 0 else 0

                # Add black pixels to the bottom and right of the image
                image = cv2.copyMakeBorder(image, 0, add_height, 0, add_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            height, width, _ = image.shape
            num_chunks = (height // self.__chunk_size) * (width // self.__chunk_size)

            for i in range(num_chunks):
                chunk_image = image[(i // (width // self.__chunk_size)) * self.__chunk_size:((i // (width // self.__chunk_size)) + 1) * self.__chunk_size,
                                    (i % (width // self.__chunk_size)) * self.__chunk_size:((i % (width // self.__chunk_size)) + 1) * self.__chunk_size]
                chunk_file_path = os.path.join(output_folder, f"{i}.jpg")
                cv2.imwrite(chunk_file_path, chunk_image)

            if self.__print_info:
                print(f"Image {os.path.basename(file_path)} chunked and saved in: {os.path.abspath(output_folder)}")

    def merge(self, input_folder, output_folder, overwrite=True):
        """
        Merge the chunked images back into a single image.

        Args:
            input_folder (str): The path to the input folder.
            output_folder (str): The path to the output folder.
            overwrite (bool, optional): Whether to overwrite the existing merged image if it already exists. Defaults to True.
        """
        # Step 1: Check if info.json exists
        info_file_path = os.path.join(input_folder, "info.json")
        if not os.path.exists(info_file_path):
            print("Warning: info.json file not found!")
            return

        # Step 2: Extract image information from info.json
        with open(info_file_path, "r") as info_file:
            info = json.load(info_file)
        original_name = info["original_name"]
        width = info["width"]
        height = info["height"]
        chunk_size = info["chunk_size"]

        # Step 3: Merge the chunked images

        num_chunks_height = math.ceil(height / chunk_size)
        num_chunks_width = math.ceil(width / chunk_size)

        merged_height = num_chunks_height * chunk_size
        merged_width = num_chunks_width * chunk_size

        # Create the merged image
        merged_image = np.zeros((merged_height, merged_width, 3), dtype=np.uint8)
        file_index = 0
        for i in range(num_chunks_height):
            for j in range(num_chunks_width):
                chunk_image_path = os.path.join(input_folder, f"{file_index}.jpg")
                chunk_image = cv2.imread(chunk_image_path)
                if chunk_image is None:
                    print(f"Warning: Failed to read chunk image: {chunk_image_path}")
                    return
                merged_image[i * chunk_size:(i + 1) * chunk_size, j * chunk_size:(j + 1) * chunk_size] = chunk_image
                file_index += 1

        # Step 4: Crop the merged image to the original size
        merged_image = merged_image[:height, :width]

        # Step 5: Save the merged image
        output_folder = output_folder if output_folder else self.__output
        merged_file_path = os.path.join(output_folder, original_name)
        if os.path.exists(merged_file_path):
            print(f"Warning: Merged image file already exists: {merged_file_path}")
            if not overwrite: return
        cv2.imwrite(merged_file_path, merged_image)

        if self.__print_info:
            print(f"Merged image saved as: {os.path.abspath(merged_file_path)}")

    