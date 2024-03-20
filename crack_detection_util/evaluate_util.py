import torch
import os
import csv
import pandas as pd

from PIL import Image
from torchvision.transforms import ToTensor

class Evaluator:

    def __init__(self, input_folder, dataset):
        '''
        The Evaluator class is used to evaluate the performance of the Semantic Segmentation models.

        Args:
            input_folder: str, the folder containing the true and prediction folders
            dataset: str, the dataset name (e.g. "crack500")
        '''
        self.input_folder = input_folder
        self.dataset = dataset
        self.train_folder = os.path.join(input_folder, 'train')
        self.val_folder = os.path.join(input_folder, 'val')

    def calculate_evaluation_metrics(self, input_folder=None, dataset=None, calculate_mae=True, calculate_miou=True, calculate_map=True):
        '''
        Calculate the evaluation metrics (MAE, mIoU, and mAP) between the true and prediction masks.
        The result will be saved in a csv file named "scores.csv" in the input folder.

        Args:
            input_folder: str, the folder containing the true and prediction folders
            calculate_mae: bool, whether to calculate MAE (default: True)
            calculate_miou: bool, whether to calculate mIoU (default: True)
            calculate_map: bool, whether to calculate mAP (default: True)
        '''
        input_folder = input_folder if input_folder else self.input_folder
        dataset = dataset if dataset else self.dataset
        evaluate_folder = os.path.join(input_folder, dataset)
        true_folder = os.path.join(evaluate_folder, 'true')
        pred_folder = os.path.join(evaluate_folder, 'prediction')

        model_scores = {}
        if calculate_mae:
            model_mae = {}
        if calculate_miou:
            model_miou = {}
        if calculate_map:
            model_map = {}

        for model_name in os.listdir(pred_folder):
            model_folder = os.path.join(pred_folder, model_name)
            if not os.path.isdir(model_folder):
                continue

            mae_sum = 0
            miou_sum = 0
            map_sum = 0
            file_count = 0

            for file_name in os.listdir(model_folder):
                pred_path = os.path.join(model_folder, file_name)
                true_path = os.path.join(true_folder, file_name)

                if not os.path.isfile(true_path):
                    continue

                true_image = Image.open(true_path)
                pred_image = Image.open(pred_path)

                transform = ToTensor()

                true_mask = transform(true_image)
                pred_mask = transform(pred_image)

                true_mask_binary = true_mask > 0
                pred_mask_binary = pred_mask > 0

                tp = torch.logical_and(true_mask_binary, pred_mask_binary).sum().item()
                tn = torch.logical_and(~true_mask_binary, ~pred_mask_binary).sum().item()
                fp = torch.logical_and(~true_mask_binary, pred_mask_binary).sum().item()
                fn = torch.logical_and(true_mask_binary, ~pred_mask_binary).sum().item()

                if calculate_mae:
                    mae = (fp + fn) / (tp + tn + fp + fn)
                    mae_sum += mae

                if calculate_miou:
                    if tp + fp + fn == 0:
                        iou = 1
                    else:
                        iou = tp / (tp + fp + fn)
                    miou_sum += iou

                if calculate_map:
                    ap = (tp + tn) / (tp + fp + tn + fn)
                    map_sum += ap

                file_count += 1

            if file_count > 0:
                if calculate_mae:
                    model_mae[model_name] = (mae_sum / file_count, file_count)
                if calculate_miou:
                    model_miou[model_name] = (miou_sum / file_count, file_count)
                if calculate_map:
                    model_map[model_name] = (map_sum / file_count, file_count)

        if calculate_mae:
            model_scores['MAE'] = model_mae
        if calculate_miou:
            model_scores['mIoU'] = model_miou
        if calculate_map:
            model_scores['mAP'] = model_map

        csv_path = os.path.join(input_folder, self.dataset + '_scores.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model Name', 'MAE', 'mIoU', 'mAP', 'File Count'])
            for model_name, (mae, mae_file_count) in model_mae.items():
                iou = model_miou.get(model_name, (0, 0))[0]
                ap = model_map.get(model_name, (0, 0))[0]
                file_count = mae_file_count
                writer.writerow([model_name, mae, iou, ap, file_count])

        print(f"Scores saved in {csv_path}")

    def merge_csv_files(self, input_folder=None):
        '''
        Merge all csv files in the input folder.
        Extract the dataset field from each file (split by the last underscore).
        Add a "dataset" field to each file with the same value.
        Output the merged file as "scores.csv" and print the output message.

        Args:
            input_folder: str, the folder containing the csv files
        '''

        input_folder = input_folder if input_folder else self.input_folder
        merged_data = pd.DataFrame()
        merged_file_path = os.path.join(input_folder, 'scores.csv')
        if os.path.isfile(merged_file_path):
            os.remove(merged_file_path)

        csv_files = [file_name for file_name in os.listdir(input_folder) if file_name.endswith('.csv')]
        for file_name in csv_files:
            dataset = file_name.rsplit('_', 1)[0]
            file_path = os.path.join(input_folder, file_name)
            data = pd.read_csv(file_path)
            data['dataset'] = dataset
            merged_data = pd.concat([merged_data, data], ignore_index=True)

        merged_data.to_csv(merged_file_path, index=False)
        print(f"Merged file saved in {merged_file_path}")