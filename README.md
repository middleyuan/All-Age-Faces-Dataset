# Goal
To develop a robotic system that can generate engaging questions about digital photos to trigger reminiscence.

# Motivation behind training the model on All-Age-Faces Dataset
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
| Age Model with 3 classes | Age Model with 8 classes | Gender Model |
|--------------------------| ------------------------ | -------------|
|           0.724          |            0.534         |    0.914     |

# Program Flow Chart
![image](https://github.com/middleyuan/All-Age-Faces-Dataset/blob/master/flow_chart.png)

# Folder Description
0.534_age_checkpoint:
* 8 classes age prediction model with 0.534 accuraccy.
* AGE_LIST = ['[0, 3]','[4, 7]','[8, 14]','[15, 24]','[25, 37]','[38, 47]','[48, 59]','[60, 100]']
0.726_age_checkpoint:
* 3 classes age prediction model with 0.726 accuraccy.
* AGE_LIST = ['[0, 28]','[29, 54]','[55, 80]']
0.914_gender_prediction_checkpoint:
* 2 classes gender prediction model with 0.914 accuraccy.
* GENDER_LIST =['F','M']
Data:
* By default, this folder contains the image files that we want to process.
output:
* By default, the python script will process an image file and output the .csv file and cropping faces of it.

# Description
Given a photo, the program can support face detection, and it will output each person's:

* face location compared with the entire photo. (The number is normalized, so it lies in [0, 1])
* face area ratio compared with the entire photo.
* age and gender prediction. 
* relation.

Afterwards, it will generate engaging question based on the information retrieved from the photo.

# Usage 

## For processing a single image file 
Pass the filename and python script will output the result to "output" folder.
```
python predict.py --filename ./Data/photo1.jpg
```

Sample output: 
```
你覺得他對你做過最浪漫的事情是什麽呢？
```

## For processing all the image files in a folder
Run the bash script and it will process all the files in "Data" folder and output result in "output" folder.
```
./run
```
or 
```
sh run
```
