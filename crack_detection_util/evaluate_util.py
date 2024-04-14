import torch
import os
import csv
import pandas as pd

from PIL import Image
from torchvision.transforms import ToTensor

class Evaluator:

    def __init__(self, input_folder, dataset=None):
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

    def calculate_evaluation_metrics(self, input_folder=None, dataset=None, beta=1):
        '''
        Calculate the evaluation metrics (MAE, mIoU, mAP, Recall, F-measure) between the true and prediction masks.
        The result will be saved in a csv file named "scores.csv" in the input folder.

        Args:
            input_folder: str, the folder containing the true and prediction folders
            dataset: str, the dataset name (e.g. "crack500")
            beta: float, the beta value for F-measure calculation (default: 1)
        '''
        input_folder = input_folder if input_folder else self.input_folder
        dataset = dataset if dataset else self.dataset
        evaluate_folder = os.path.join(input_folder, dataset)
        true_folder = os.path.join(evaluate_folder, 'true')
        pred_folder = os.path.join(evaluate_folder, 'prediction')

        model_scores = {}
        model_recall = {}
        model_f_measure = {}

        total_files = 0

        for model_name in os.listdir(pred_folder):
            model_folder = os.path.join(pred_folder, model_name)
            if not os.path.isdir(model_folder):
                continue

            mae_sum = 0
            miou_sum = 0
            map_sum = 0
            recall_sum = 0
            f_measure_sum = 0
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

                true_mask_binary = true_mask > 0.5
                pred_mask_binary = pred_mask > 0.5

                # Dilate the true masks
                dilate_radius = 1
                dilated_mask = torch.nn.functional.max_pool2d(true_mask_binary.float(), kernel_size=dilate_radius*2+1, stride=1, padding=dilate_radius)
                true_mask_binary_fixed = dilated_mask > 0.5

                tp = torch.logical_and(true_mask_binary_fixed, pred_mask_binary).sum().item()
                tn = torch.logical_and(~true_mask_binary, ~pred_mask_binary).sum().item()
                fp = torch.logical_and(~true_mask_binary_fixed, pred_mask_binary).sum().item()
                fn = torch.logical_and(true_mask_binary, ~pred_mask_binary).sum().item()

                # MAE
                mae = (fp + fn) / (tp + tn + fp + fn)
                mae_sum += mae

                # Recall
                if tp + fn == 0:
                    recall = 1
                else:
                    recall = tp / (tp + fn)
                recall_sum += recall

                # Mean IoU
                if tp + fp + fn == 0:
                    iou = 1
                else:
                    iou = tp / (tp + fp + fn)
                miou_sum += iou

                # Precision
                if tp + fp == 0:
                    ap = 1
                else:
                    ap = tp / (tp + fp)
                map_sum += ap

                # F-measure
                if tp + fp + fn == 0 or tp == 0:
                    f_measure = 0
                else:
                    # precision = tp / (tp + fp)
                    # recall = tp / (tp + fn)
                    # f_measure = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
                    f_measure = (2 * tp) / (2 * tp + fp + fn)
                f_measure_sum += f_measure

                file_count += 1

            if file_count > 0:
                model_scores[model_name] = {
                    'MAE': (mae_sum / file_count),
                    'mIoU': (miou_sum / file_count),
                    'mAP': (map_sum / file_count),
                    'Recall': (recall_sum / file_count),
                    'F-measure': (f_measure_sum / file_count)
            }

            total_files = file_count

        csv_path = os.path.join(input_folder, dataset + '_scores.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Dataset', 'Total Files', 'Model Name', 'MAE', 'mIoU', 'mAP', 'Recall', 'F-measure'])
            for model_name, scores in model_scores.items():
                writer.writerow([dataset, total_files, model_name, scores['MAE'], scores['mIoU'], scores['mAP'], scores['Recall'], scores['F-measure']])

        print(f"Scores saved in {csv_path}")

    def merge_csv_files(self, input_folder=None):

        '''
        Merge all csv files in the input folder.
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
            file_path = os.path.join(input_folder, file_name)
            data = pd.read_csv(file_path)
            merged_data = pd.concat([merged_data, data], ignore_index=True)

        merged_data.to_csv(merged_file_path, index=False)
        print(f"Merged file saved in {merged_file_path}")

    def evaluate_all_datasets(self, input_folder=None):
        '''
        Evaluate all datasets in the input folder and merge the results into a single CSV file.

        Args:
            input_folder: str, the folder containing the datasets
        '''
        input_folder = input_folder if input_folder else self.input_folder

        evaluator = Evaluator(input_folder, '')
        dataset_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]
        
        for dataset_folder in dataset_folders:
            evaluator.calculate_evaluation_metrics(input_folder=input_folder, dataset=dataset_folder)
        
        evaluator.merge_csv_files(input_folder=input_folder)
