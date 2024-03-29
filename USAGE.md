## Quick Start

### For image utilities

Here is an example of how I am using this in the actual project.

```python
import os

from predict import predict
from crack_detection_util.image_util import MyImage

def main():
    folder = "./predict"

    mi = MyImage(model_name='maskrcnn',chunk_size=512, folder=folder)

    mi.clear_cache()
    mi.batch_chunk_images()
    mi.batch_predict_images(predict=predict)
    mi.batch_merge_images()

if __name__ == '__main__':
    main()

# The path structure is as follows
# .root/
#     predict.py -- the predict function
#     batch_predict.py --this file
#     /predict
#         /input -- put the images in this folder to be predicted
#         /cache -- this folder is automatically generated
#         /output -- this folder is automatically generated, get the outputs here
#     ... -- other dependencies for predict
```

### For evaluation utilities

Here is an example of how I am using this in the actual project.

```python
from crack_detection_util.evaluate_util import Evaluator

data = "example/evaluation"
dataset = "PCL7"
ev = Evaluator(input_folder=data, dataset=dataset)

ev.calculate_evaluation_metrics()
ev.merge_csv_files()

# The path structure is as follows
# .root/
#     example.ipynb -- this file
#     /example
#         /evaluation
#             /PCL7 -- name of dataset, put all masks files here as following rules
#                 /true -- put all true mask images(*.jpg) here
#                 /prediction -- put all prediction mask images(*.jpg) here, divided by prediction models
#                     /UNet
#                     /NestedUNet
#                     ...
#             /... -- put more than 1 datasets here is ok, but you need to call them in your program
#             PCL7_scores.csv -- this file is automatically generated including evaluation metrics for a single dataset and all prediction models
#             scores.csv -- this file is automatically generated including evaluation metrics for all datasets and all prediction models (merged)
```