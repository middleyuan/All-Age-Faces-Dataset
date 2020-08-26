# Goal
Do age and gender prediction on pictures.

# Motivation
Nowadays, I find few age and gender prediction models are trained on Asian dataset.
It will cause a drop of accuracy when the targets we went to predict are Asian people.
So, I would try to train a model based on dataset containing mostly Asian.
Please refer the link to see the detail of the dataset: All-Age-Faces Dataset
(https://github.com/JingchunCheng/All-Age-Faces-Dataset) 

# Pre-process data for training
You can preprocess All-Age-Faces Dataset by using datapreproc.py. Regarding the age, it will re-label the data to 
8 intervals. That is, [0, 3], [4, 7], [8, 14], [15, 24], [25, 37], [38, 47], [48, 59], [60, -]. 
Also, I split the data into training data (10137) and validation data (3185).
Finally, datapreproc.py process the both age and gender data set and save them as TFRecord.
You can check the TFRecord of age and TFRecord of gender in age directory and gender directory repectively.

# Accuracy
|   Age Model   | Gender Model |
| ------------- | -------------|
|     0.534     |    0.914     |

# To-do list
* Release pre-trained checkpoints.
* Release predict module that can be used to predict age and gender of faces.
