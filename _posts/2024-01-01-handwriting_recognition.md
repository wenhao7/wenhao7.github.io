---
layout: post
title:  "Handwriting Recognition"
date:   2024-01-01 22:31:08 +0800
category: [data_wrangling, visualization, deep_learning]
tag: [numpy, pandas, cv2, matplotlib, tensorflow, nlp, computer_vision, flask]
summary: "In this notebook we build and deploy a flask web application to detect and recognise handwritten words from an image."
---

## Contents
1. [Overview](#1)
2. [Contour Detection](#2)
3. [Image Recognition](#3)
4. [Deployment](#4)
5. [Final Thought](#5)

***

<a id = '1'></a>
## 1. Overview 
In this notebook we will build and deploy a flask web application to detect and recognise handwritten words from an image.

***

<a id = '2'></a>
## 2. Contour Detection
This model will make a few assumptions about the input images:
* Handwriting is neat and font resembles prints (not cursive)
* Writings are formatted in clearly discernable rows of sentences (no overlapping words or words rotated at different angles, like writing on lined paper))
<br>

With these assumptions the model can "read" the image similar to how a person would, from left to right starting from the top row to the bottom row. The model will recognise and predict each alphabet in a word individually and attempt to group alphabets belonging to the same word together.

#### Differentiate the separate rows of words
The following code detects the top and bottom edges of each row of words.


```python
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Untitled_Artwork.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(img)
plt.show()

# Detect left and right edges of alphabets
edges = cv2.Canny(img,100,200)
horizontal_sum = np.sum(edges, axis = 1)
horizontal_sum = horizontal_sum != 0
horizontal_changes = np.logical_xor(horizontal_sum[1:], horizontal_sum[:-1])
horizontal_change_pts = np.nonzero(horizontal_changes)[0]

# Detect top and bottom edges of each row of words
vertical_change_pts = {}
for i in range(0, len(horizontal_change_pts), 2):
    start, end = horizontal_change_pts[i], horizontal_change_pts[i + 1]
    vertical_sum = np.sum(edges[start:end], axis = 0)
    vertical_sum = vertical_sum != 0
    vertical_changes = np.logical_xor(vertical_sum[1:], vertical_sum[:-1])
    vertical_change_pts[start] = (start, end, np.nonzero(vertical_changes)[0])

plt.imshow(img)
for change in horizontal_change_pts:
    plt.axhline(change)
for line in vertical_change_pts:
    for change in vertical_change_pts[line][2]:
        plt.axvline(change, 1- vertical_change_pts[line][0]/img.shape[0], 1- vertical_change_pts[line][1]/img.shape[0])
plt.show()
```


    
![png](/images/handwriting_recognition/output_1_0.png)
    



    
![png](/images/handwriting_recognition/output_1_1.png)
    


#### Detect contours of each alphabet and obtain their bounding box coordinates
The following code finds the bounding box for each alphabet that will be grouped together to form words after prediction. The variable "close_dist" define the distance between alphabets to determine if they belong to the same word.


```python
image = cv2.imread('Untitled_Artwork.png')
original = image.copy()
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours, obtain bounding boxes
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

boxes = []
close_dist = 8
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    
    # x1, y1 coordinates of bottom left, x2, y2 coordinates of top right
    x1, y1, x2, y2 = max(0, x), min(image.shape[0], y+h), min(image.shape[1], x+w), max(0,y)
    boxes.append([x1, y1, x2, y2])
boxes.sort(key = lambda x:x[3])

for box in boxes:
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (36,255,12), 2)   
plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x1f17c754370>




    
![png](/images/handwriting_recognition/output_3_1.png)
    



```python
# Function to check if two bounding boxes are overlapping
def isRectangleOverlap(rec1, rec2):
    return not (rec1[2] <= rec2[0] or  # left
                    rec1[3] >= rec2[1] or  # bottom
                    rec1[0] >= rec2[2] or  # right
                    rec1[1] <= rec2[3])    # top

# Function to assign individual bounding boxes to their respective row number
def binY(horizontal_change_pts, boxes):
    bins, ind = [], 0
    for i in range(0, len(horizontal_change_pts), 2):
        j = i+1
        bins.append((ind, horizontal_change_pts[i], horizontal_change_pts[j]))
        ind += 1
    
    for i in range(len(boxes)):
        center = (boxes[i][1] + boxes[i][3]) // 2
        for j in range(len(bins)):
            if bins[j][1] <= center <= bins[j][2]:
                boxes[i].append(bins[j][0])
    boxes.sort(key = lambda x: (x[-1], x[0]))
    return bins, boxes
```

With the coordinate information of the bounding boxes the overlapping boxes can be grouped together to form word level bounding boxes. An additional step along the way would be to identify the dots from the "i" and "j" characters to group them into the same alphabet level bounding box for prediction.


```python
image = cv2.imread('Untitled_Artwork.png', cv2.IMREAD_GRAYSCALE)
single_bounding_boxes_filepath = 'single_bounding_boxes\\'

# Function to include the dots of i and j in the alphabet's bounding box
def combineIJdots(boxes):
    combined_boxes = []
    prev = boxes[0]
    for i, box in enumerate(boxes):
        coords, word_bin = box[:4], box[-1]
        overlap_coords = [max(0, coords[0]),
                          min(image.shape[0], coords[1] + 3),
                          min(image.shape[1], coords[2] + 1),
                          max(0, coords[3] - 3),
                          word_bin]
        box1 = prev
        box2 = overlap_coords
        overlap = isRectangleOverlap(prev, overlap_coords)
        if overlap:
            prev = [min(box1[0], box2[0]),
                    max(box1[1], box2[1]),
                    max(box1[2], box2[2]),
                    min(box1[3], box2[3]),
                    word_bin]
        else:
            combined_boxes.append(prev)
            prev = overlap_coords
    
    if overlap:
        combined_boxes.append(prev)
    else:
        combined_boxes.append(overlap_coords)
    
    return combined_boxes

# Function to combine bounding boxes at alphabet level to word level
def combineBoxes(boxes):
    combined_boxes = []
    file_paths = []
    word_count, curr_bin = 0, -1
    for i, box in enumerate(boxes):
        coords, word_bin = box[:4], box[-1]
        overlap_coords = [max(0, coords[0]),
                          min(image.shape[0], coords[1] + close_dist//2),
                          min(image.shape[1], coords[2] + close_dist),
                          max(0, coords[3] - close_dist//2),
                          word_count]
        if curr_bin < word_bin:
            combined_boxes.append(overlap_coords)
            word_count += 1
            curr_bin += 1
        else:
            box1 = combined_boxes[word_count-1]
            box2 = overlap_coords
            overlap = isRectangleOverlap(box1, box2)
            if overlap:
                combined_boxes[word_count - 1] = [min(box1[0], box2[0]),
                                     max(box1[1], box2[1]),
                                     max(box1[2], box2[2]),
                                     min(box1[3], box2[3]),
                                     word_count]
            else:
                combined_boxes.append(overlap_coords)
                word_count += 1
                
        # Save images from individual bounding boxes
        boundaries = image[coords[3]:coords[1], coords[0]:coords[2]]
        single_file_path = str(single_bounding_boxes_filepath) + 'ROI_' + str(i) + '_' + str(word_count) + '.png'
        padded_image = cv2.copyMakeBorder(boundaries, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value = 255)
        cv2.imwrite(os.getcwd() + '\\' + single_file_path, padded_image)
        #cv2.imwrite(os.getcwd() + '\\' + single_file_path, boundaries)
        file_paths.append((word_count, single_file_path))
        
    return combined_boxes, file_paths


bins, boxes = binY(horizontal_change_pts, boxes)
boxes = combineIJdots(boxes)
combined_boxes, file_paths = combineBoxes(boxes)

image = cv2.imread('Untitled_Artwork.png', cv2.IMREAD_GRAYSCALE)
for box in combined_boxes:
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (36,255,12), 2) 
plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x1f17c7e2880>




    
![png](/images/handwriting_recognition/output_6_1.png)
    


***

<a id = '3'></a>
## 3. Image Recognition
For image recognition we will be using [this dataset](https://www.kaggle.com/dhruvildave/english-handwritten-characters-dataset) from Kaggle to build a CNN that will predict for individual alphabets.


```python
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation
from keras import optimizers
from keras.models import save_model, load_model

# Variables
train_ = 0.95
validation_ = 0.05

batch_size = 32
image_width = 128
image_height = 128
img_size = (image_height, image_width)

np.random.seed(33)
tf.random.set_seed(33)

epochs = 100
```


```python
# Read Dataset
base_dir = os.getcwd()
base_path = base_dir + '\\data'
df = pd.read_csv(base_dir + "\\data\\english.csv")
print("Before \n",df.head())
df = df.loc[df['label'] >= 'A'] # Exclude numerical characters, keep only A-Z and a-z
print("After \n",df.head())
```

    Before 
                     image label
    0  Img/img001-001.png     0
    1  Img/img001-002.png     0
    2  Img/img001-003.png     0
    3  Img/img001-004.png     0
    4  Img/img001-005.png     0
    After 
                       image label
    550  Img/img011-001.png     A
    551  Img/img011-002.png     A
    552  Img/img011-003.png     A
    553  Img/img011-004.png     A
    554  Img/img011-005.png     A
    


```python
unique_labels = len(df.label.unique())
image_path = base_dir + base_path + '\\Img'

training_samples = df.groupby('label').apply(lambda s: s.sample(int(len(s) * train_)))
validation_samples = df[~df.image.isin(training_samples.image)]
print(f"Number of Training Samples found : {len(training_samples)}" )
print(f"Number of Validation Samples found : {len(validation_samples)}" )
```

    Number of Training Samples found : 2704
    Number of Validation Samples found : 156
    


```python
# Image generator
train_data_generator = ImageDataGenerator(
            rescale=1/255,
            shear_range=0.1,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,)
data_generator = ImageDataGenerator(rescale = 1/255)

training_data_frame = train_data_generator.flow_from_dataframe( 
                                            dataframe = training_samples, 
                                            directory = base_path,
                                            x_col = 'image',
                                            y_col = 'label',
                                            target_size = img_size,
                                            class_mode = 'categorical')
validation_data_frame = data_generator.flow_from_dataframe( 
                                            dataframe = validation_samples, 
                                            directory = base_path,
                                            x_col = 'image',
                                            y_col = 'label',
                                            target_size = img_size,
                                            class_mode = 'categorical')
```

    Found 2704 validated image filenames belonging to 52 classes.
    Found 156 validated image filenames belonging to 52 classes.
    


```python
# Building the model

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = [image_height,image_width,3]))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(unique_labels, activation = 'softmax'))

opt = optimizers.Adam(epsilon = 0.01)
model.compile(
            optimizer=opt,
            loss= 'categorical_crossentropy',
            metrics = ['accuracy'])
```


```python
#Training
history = model.fit(training_data_frame, validation_data = validation_data_frame, 
                    steps_per_epoch = len(training_samples)//32 + 1,
                    epochs = epochs,
                    validation_steps = len(validation_samples)//32 + 1,
                    verbose = 1
                )
```

    Epoch 1/100
    85/85 [==============================] - 62s 514ms/step - loss: 3.9812 - accuracy: 0.0201 - val_loss: 3.9530 - val_accuracy: 0.0000e+00
    Epoch 2/100
    85/85 [==============================] - 31s 373ms/step - loss: 3.9511 - accuracy: 0.0179 - val_loss: 3.9500 - val_accuracy: 0.0256
    Epoch 3/100
    85/85 [==============================] - 31s 372ms/step - loss: 3.9519 - accuracy: 0.0164 - val_loss: 3.9459 - val_accuracy: 0.0192
    Epoch 4/100
    85/85 [==============================] - 31s 369ms/step - loss: 3.9472 - accuracy: 0.0275 - val_loss: 3.9379 - val_accuracy: 0.0449
    Epoch 5/100
    85/85 [==============================] - 31s 370ms/step - loss: 3.9364 - accuracy: 0.0321 - val_loss: 3.8607 - val_accuracy: 0.0321
    Epoch 6/100
    85/85 [==============================] - 31s 373ms/step - loss: 3.8452 - accuracy: 0.0472 - val_loss: 3.4647 - val_accuracy: 0.1410
    Epoch 7/100
    85/85 [==============================] - 31s 373ms/step - loss: 3.6368 - accuracy: 0.1057 - val_loss: 3.0245 - val_accuracy: 0.2756
    Epoch 8/100
    85/85 [==============================] - 31s 371ms/step - loss: 3.4227 - accuracy: 0.1268 - val_loss: 2.8402 - val_accuracy: 0.3013
    Epoch 9/100
    85/85 [==============================] - 31s 369ms/step - loss: 3.2149 - accuracy: 0.1687 - val_loss: 2.4654 - val_accuracy: 0.3333
    Epoch 10/100
    85/85 [==============================] - 31s 370ms/step - loss: 3.0666 - accuracy: 0.2012 - val_loss: 2.3074 - val_accuracy: 0.4295
    Epoch 11/100
    85/85 [==============================] - 33s 391ms/step - loss: 2.8731 - accuracy: 0.2395 - val_loss: 2.1982 - val_accuracy: 0.4808
    Epoch 12/100
    85/85 [==============================] - 35s 422ms/step - loss: 2.7432 - accuracy: 0.2585 - val_loss: 2.1429 - val_accuracy: 0.4167
    Epoch 13/100
    85/85 [==============================] - 35s 412ms/step - loss: 2.6434 - accuracy: 0.2910 - val_loss: 2.0547 - val_accuracy: 0.5192
    Epoch 14/100
    85/85 [==============================] - 34s 402ms/step - loss: 2.5168 - accuracy: 0.3001 - val_loss: 1.8342 - val_accuracy: 0.5128
    Epoch 15/100
    85/85 [==============================] - 34s 401ms/step - loss: 2.2935 - accuracy: 0.3699 - val_loss: 1.5608 - val_accuracy: 0.5705
    Epoch 16/100
    85/85 [==============================] - 35s 415ms/step - loss: 2.1660 - accuracy: 0.3952 - val_loss: 1.4810 - val_accuracy: 0.5705
    Epoch 17/100
    85/85 [==============================] - 35s 415ms/step - loss: 2.0415 - accuracy: 0.4228 - val_loss: 1.2098 - val_accuracy: 0.6731
    Epoch 18/100
    85/85 [==============================] - 37s 443ms/step - loss: 1.8782 - accuracy: 0.4544 - val_loss: 1.1975 - val_accuracy: 0.6410
    Epoch 19/100
    85/85 [==============================] - 35s 414ms/step - loss: 1.7191 - accuracy: 0.4870 - val_loss: 1.1556 - val_accuracy: 0.6859
    Epoch 20/100
    85/85 [==============================] - 31s 372ms/step - loss: 1.5540 - accuracy: 0.5574 - val_loss: 0.8923 - val_accuracy: 0.6923
    Epoch 21/100
    85/85 [==============================] - 31s 371ms/step - loss: 1.4708 - accuracy: 0.5855 - val_loss: 0.9662 - val_accuracy: 0.8013
    Epoch 22/100
    85/85 [==============================] - 31s 370ms/step - loss: 1.4229 - accuracy: 0.5860 - val_loss: 0.9162 - val_accuracy: 0.7500
    Epoch 23/100
    85/85 [==============================] - 31s 369ms/step - loss: 1.4211 - accuracy: 0.5808 - val_loss: 0.8532 - val_accuracy: 0.7500
    Epoch 24/100
    85/85 [==============================] - 31s 371ms/step - loss: 1.3173 - accuracy: 0.5952 - val_loss: 0.9616 - val_accuracy: 0.7308
    Epoch 25/100
    85/85 [==============================] - 31s 373ms/step - loss: 1.2238 - accuracy: 0.6290 - val_loss: 0.7518 - val_accuracy: 0.7821
    Epoch 26/100
    85/85 [==============================] - 38s 450ms/step - loss: 1.2281 - accuracy: 0.6336 - val_loss: 0.8121 - val_accuracy: 0.7821
    Epoch 27/100
    85/85 [==============================] - 32s 378ms/step - loss: 1.1671 - accuracy: 0.6334 - val_loss: 0.7623 - val_accuracy: 0.7949
    Epoch 28/100
    85/85 [==============================] - 32s 382ms/step - loss: 1.1649 - accuracy: 0.6284 - val_loss: 0.6636 - val_accuracy: 0.7756
    Epoch 29/100
    85/85 [==============================] - 32s 382ms/step - loss: 1.1412 - accuracy: 0.6640 - val_loss: 0.8688 - val_accuracy: 0.7500
    Epoch 30/100
    85/85 [==============================] - 35s 420ms/step - loss: 1.0698 - accuracy: 0.6746 - val_loss: 0.6431 - val_accuracy: 0.8141
    Epoch 31/100
    85/85 [==============================] - 36s 430ms/step - loss: 0.9960 - accuracy: 0.6866 - val_loss: 0.7038 - val_accuracy: 0.7885
    Epoch 32/100
    85/85 [==============================] - 33s 388ms/step - loss: 0.9823 - accuracy: 0.7071 - val_loss: 0.5744 - val_accuracy: 0.8141
    Epoch 33/100
    85/85 [==============================] - 33s 393ms/step - loss: 0.9299 - accuracy: 0.6998 - val_loss: 0.7822 - val_accuracy: 0.8141
    Epoch 34/100
    85/85 [==============================] - 36s 430ms/step - loss: 0.9344 - accuracy: 0.7081 - val_loss: 0.4987 - val_accuracy: 0.8590
    Epoch 35/100
    85/85 [==============================] - 32s 380ms/step - loss: 0.8488 - accuracy: 0.7412 - val_loss: 0.7139 - val_accuracy: 0.7821
    Epoch 36/100
    85/85 [==============================] - 34s 400ms/step - loss: 0.8570 - accuracy: 0.7239 - val_loss: 0.5269 - val_accuracy: 0.8526
    Epoch 37/100
    85/85 [==============================] - 34s 409ms/step - loss: 0.8067 - accuracy: 0.7397 - val_loss: 0.4965 - val_accuracy: 0.8782
    Epoch 38/100
    85/85 [==============================] - 32s 376ms/step - loss: 0.8118 - accuracy: 0.7309 - val_loss: 0.5196 - val_accuracy: 0.8462
    Epoch 39/100
    85/85 [==============================] - 32s 379ms/step - loss: 0.8282 - accuracy: 0.7456 - val_loss: 0.4182 - val_accuracy: 0.8590
    Epoch 40/100
    85/85 [==============================] - 32s 385ms/step - loss: 0.8181 - accuracy: 0.7507 - val_loss: 0.6560 - val_accuracy: 0.8526
    Epoch 41/100
    85/85 [==============================] - 30s 354ms/step - loss: 0.7603 - accuracy: 0.7634 - val_loss: 0.5705 - val_accuracy: 0.8526
    Epoch 42/100
    85/85 [==============================] - 30s 353ms/step - loss: 0.7564 - accuracy: 0.7441 - val_loss: 0.6256 - val_accuracy: 0.8333
    Epoch 43/100
    85/85 [==============================] - 30s 352ms/step - loss: 0.6911 - accuracy: 0.7898 - val_loss: 0.5376 - val_accuracy: 0.8462
    Epoch 44/100
    85/85 [==============================] - 30s 360ms/step - loss: 0.6788 - accuracy: 0.7913 - val_loss: 0.3983 - val_accuracy: 0.9167
    Epoch 45/100
    85/85 [==============================] - 30s 351ms/step - loss: 0.6876 - accuracy: 0.7855 - val_loss: 0.4095 - val_accuracy: 0.8782
    Epoch 46/100
    85/85 [==============================] - 30s 352ms/step - loss: 0.6923 - accuracy: 0.7801 - val_loss: 0.5367 - val_accuracy: 0.8782
    Epoch 47/100
    85/85 [==============================] - 30s 352ms/step - loss: 0.6636 - accuracy: 0.7758 - val_loss: 0.5154 - val_accuracy: 0.8846
    Epoch 48/100
    85/85 [==============================] - 30s 353ms/step - loss: 0.6696 - accuracy: 0.7958 - val_loss: 0.5643 - val_accuracy: 0.8590
    Epoch 49/100
    85/85 [==============================] - 34s 402ms/step - loss: 0.6097 - accuracy: 0.8032 - val_loss: 0.5509 - val_accuracy: 0.8974
    Epoch 50/100
    85/85 [==============================] - 34s 406ms/step - loss: 0.7247 - accuracy: 0.7613 - val_loss: 0.5525 - val_accuracy: 0.8205
    Epoch 51/100
    85/85 [==============================] - 31s 364ms/step - loss: 0.6036 - accuracy: 0.8101 - val_loss: 0.5755 - val_accuracy: 0.8397
    Epoch 52/100
    85/85 [==============================] - 29s 350ms/step - loss: 0.6231 - accuracy: 0.8010 - val_loss: 0.4373 - val_accuracy: 0.8846
    Epoch 53/100
    85/85 [==============================] - 29s 350ms/step - loss: 0.5865 - accuracy: 0.8187 - val_loss: 0.5371 - val_accuracy: 0.8718
    Epoch 54/100
    85/85 [==============================] - 30s 354ms/step - loss: 0.5810 - accuracy: 0.8029 - val_loss: 0.4212 - val_accuracy: 0.8718
    Epoch 55/100
    85/85 [==============================] - 30s 357ms/step - loss: 0.5295 - accuracy: 0.8304 - val_loss: 0.5073 - val_accuracy: 0.8462
    Epoch 56/100
    85/85 [==============================] - 30s 354ms/step - loss: 0.5674 - accuracy: 0.8185 - val_loss: 0.4623 - val_accuracy: 0.8654
    Epoch 57/100
    85/85 [==============================] - 30s 351ms/step - loss: 0.5253 - accuracy: 0.8206 - val_loss: 0.4306 - val_accuracy: 0.8782
    Epoch 58/100
    85/85 [==============================] - 30s 357ms/step - loss: 0.5299 - accuracy: 0.8313 - val_loss: 0.6592 - val_accuracy: 0.8397
    Epoch 59/100
    85/85 [==============================] - 32s 377ms/step - loss: 0.5397 - accuracy: 0.8088 - val_loss: 0.4327 - val_accuracy: 0.8526
    Epoch 60/100
    85/85 [==============================] - 32s 376ms/step - loss: 0.5217 - accuracy: 0.8294 - val_loss: 0.4744 - val_accuracy: 0.8462
    Epoch 61/100
    85/85 [==============================] - 31s 374ms/step - loss: 0.5387 - accuracy: 0.8261 - val_loss: 0.4410 - val_accuracy: 0.8718
    Epoch 62/100
    85/85 [==============================] - 32s 378ms/step - loss: 0.4848 - accuracy: 0.8387 - val_loss: 0.4461 - val_accuracy: 0.9038
    Epoch 63/100
    85/85 [==============================] - 31s 374ms/step - loss: 0.5062 - accuracy: 0.8307 - val_loss: 0.5446 - val_accuracy: 0.8910
    Epoch 64/100
    85/85 [==============================] - 32s 376ms/step - loss: 0.5004 - accuracy: 0.8152 - val_loss: 0.2339 - val_accuracy: 0.9231
    Epoch 65/100
    85/85 [==============================] - 31s 375ms/step - loss: 0.5167 - accuracy: 0.8187 - val_loss: 0.3944 - val_accuracy: 0.9103
    Epoch 66/100
    85/85 [==============================] - 32s 378ms/step - loss: 0.4816 - accuracy: 0.8252 - val_loss: 0.4764 - val_accuracy: 0.8718
    Epoch 67/100
    85/85 [==============================] - 31s 370ms/step - loss: 0.5028 - accuracy: 0.8266 - val_loss: 0.4197 - val_accuracy: 0.8782
    Epoch 68/100
    85/85 [==============================] - 31s 371ms/step - loss: 0.4944 - accuracy: 0.8401 - val_loss: 0.2939 - val_accuracy: 0.9359
    Epoch 69/100
    85/85 [==============================] - 31s 373ms/step - loss: 0.4436 - accuracy: 0.8472 - val_loss: 0.6392 - val_accuracy: 0.8590
    Epoch 70/100
    85/85 [==============================] - 31s 374ms/step - loss: 0.4689 - accuracy: 0.8399 - val_loss: 0.3619 - val_accuracy: 0.9103
    Epoch 71/100
    85/85 [==============================] - 31s 370ms/step - loss: 0.3994 - accuracy: 0.8602 - val_loss: 0.4681 - val_accuracy: 0.9103
    Epoch 72/100
    85/85 [==============================] - 32s 375ms/step - loss: 0.4376 - accuracy: 0.8501 - val_loss: 0.4706 - val_accuracy: 0.8974
    Epoch 73/100
    85/85 [==============================] - 31s 372ms/step - loss: 0.4126 - accuracy: 0.8619 - val_loss: 0.5546 - val_accuracy: 0.8654
    Epoch 74/100
    85/85 [==============================] - 32s 380ms/step - loss: 0.4574 - accuracy: 0.8640 - val_loss: 0.3918 - val_accuracy: 0.8974
    Epoch 75/100
    85/85 [==============================] - 30s 361ms/step - loss: 0.4624 - accuracy: 0.8518 - val_loss: 0.3877 - val_accuracy: 0.8718
    Epoch 76/100
    85/85 [==============================] - 30s 359ms/step - loss: 0.4021 - accuracy: 0.8634 - val_loss: 0.4838 - val_accuracy: 0.8654
    Epoch 77/100
    85/85 [==============================] - 30s 357ms/step - loss: 0.4577 - accuracy: 0.8441 - val_loss: 0.4764 - val_accuracy: 0.8974
    Epoch 78/100
    85/85 [==============================] - 30s 358ms/step - loss: 0.4188 - accuracy: 0.8728 - val_loss: 0.3990 - val_accuracy: 0.8782
    Epoch 79/100
    85/85 [==============================] - 30s 357ms/step - loss: 0.3979 - accuracy: 0.8535 - val_loss: 0.3619 - val_accuracy: 0.9231
    Epoch 80/100
    85/85 [==============================] - 30s 358ms/step - loss: 0.3917 - accuracy: 0.8658 - val_loss: 0.3967 - val_accuracy: 0.8782
    Epoch 81/100
    85/85 [==============================] - 30s 357ms/step - loss: 0.3551 - accuracy: 0.8736 - val_loss: 0.4608 - val_accuracy: 0.8718
    Epoch 82/100
    85/85 [==============================] - 30s 354ms/step - loss: 0.4172 - accuracy: 0.8515 - val_loss: 0.4309 - val_accuracy: 0.8718
    Epoch 83/100
    85/85 [==============================] - 30s 354ms/step - loss: 0.3690 - accuracy: 0.8688 - val_loss: 0.3292 - val_accuracy: 0.9167
    Epoch 84/100
    85/85 [==============================] - 30s 355ms/step - loss: 0.3661 - accuracy: 0.8648 - val_loss: 0.5464 - val_accuracy: 0.8654
    Epoch 85/100
    85/85 [==============================] - 29s 351ms/step - loss: 0.3983 - accuracy: 0.8582 - val_loss: 0.3977 - val_accuracy: 0.8910
    Epoch 86/100
    85/85 [==============================] - 29s 350ms/step - loss: 0.3888 - accuracy: 0.8570 - val_loss: 0.4739 - val_accuracy: 0.8718
    Epoch 87/100
    85/85 [==============================] - 30s 353ms/step - loss: 0.3790 - accuracy: 0.8646 - val_loss: 0.4252 - val_accuracy: 0.9038
    Epoch 88/100
    85/85 [==============================] - 29s 345ms/step - loss: 0.3929 - accuracy: 0.8637 - val_loss: 0.4648 - val_accuracy: 0.8590
    Epoch 89/100
    85/85 [==============================] - 29s 344ms/step - loss: 0.3803 - accuracy: 0.8703 - val_loss: 0.4910 - val_accuracy: 0.8910
    Epoch 90/100
    85/85 [==============================] - 29s 345ms/step - loss: 0.4060 - accuracy: 0.8546 - val_loss: 0.3257 - val_accuracy: 0.8974
    Epoch 91/100
    85/85 [==============================] - 29s 347ms/step - loss: 0.3657 - accuracy: 0.8762 - val_loss: 0.4718 - val_accuracy: 0.8782
    Epoch 92/100
    85/85 [==============================] - 29s 344ms/step - loss: 0.3884 - accuracy: 0.8645 - val_loss: 0.3908 - val_accuracy: 0.8718
    Epoch 93/100
    85/85 [==============================] - 29s 346ms/step - loss: 0.4028 - accuracy: 0.8672 - val_loss: 0.4771 - val_accuracy: 0.8910
    Epoch 94/100
    85/85 [==============================] - 29s 347ms/step - loss: 0.3613 - accuracy: 0.8714 - val_loss: 0.4458 - val_accuracy: 0.8846
    Epoch 95/100
    85/85 [==============================] - 29s 345ms/step - loss: 0.3365 - accuracy: 0.8779 - val_loss: 0.3649 - val_accuracy: 0.9167
    Epoch 96/100
    85/85 [==============================] - 29s 345ms/step - loss: 0.3127 - accuracy: 0.8895 - val_loss: 0.3974 - val_accuracy: 0.9103
    Epoch 97/100
    85/85 [==============================] - 29s 345ms/step - loss: 0.3612 - accuracy: 0.8720 - val_loss: 0.3494 - val_accuracy: 0.9231
    Epoch 98/100
    85/85 [==============================] - 29s 345ms/step - loss: 0.3466 - accuracy: 0.8732 - val_loss: 0.3087 - val_accuracy: 0.8974
    Epoch 99/100
    85/85 [==============================] - 29s 345ms/step - loss: 0.3465 - accuracy: 0.8659 - val_loss: 0.4159 - val_accuracy: 0.9103
    Epoch 100/100
    85/85 [==============================] - 30s 353ms/step - loss: 0.3174 - accuracy: 0.8882 - val_loss: 0.4890 - val_accuracy: 0.8654
    


```python
#Evaluation
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Model saving
model.save('handwriting_recognition.h5')
```


    
![png](/images/handwriting_recognition/output_14_0.png)
    


During the training phase we observe approximately 85~90% validation accuracy, suggesting that the model will likely misclassify at least one alphabet in most words which may reduce the effectiveness of the model.

#### Predicting words within an image
Next we shall see how well the prediction works on the image seen in Section 2 of this notebook


```python
pred_df = pd.DataFrame(file_paths, columns = ['word', 'fp'])
pred_df['fake_class'] = '0'

base_dir = os.getcwd()
base_path = base_dir
pred_data_frame = data_generator.flow_from_dataframe( 
                                            dataframe = pred_df, 
                                            directory = base_path,
                                            x_col = 'fp',
                                            y_col = 'fake_class',
                                            target_size = img_size,
                                            class_mode = 'categorical',
                                            shuffle = False)

model = load_model('handwriting_recognition.h5')

pred = model.predict(pred_data_frame, steps = len(file_paths)//32 + 1)
pred_classes = pred.argmax(axis = 1)
pred_df['prediction'] = pred_classes
pred_df.head()
```

    Found 35 validated image filenames belonging to 1 classes.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>fp</th>
      <th>fake_class</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>single_bounding_boxes\ROI_0_1.png</td>
      <td>0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>single_bounding_boxes\ROI_1_1.png</td>
      <td>0</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>single_bounding_boxes\ROI_2_1.png</td>
      <td>0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>single_bounding_boxes\ROI_3_2.png</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>single_bounding_boxes\ROI_4_2.png</td>
      <td>0</td>
      <td>46</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert numerical label to alphabets
classDict = training_data_frame.class_indices
classDict = dict((v, k) for k, v in classDict.items())

pred_df['predicted_labels'] = pred_classes
pred_df['predicted_labels'] = pred_df['predicted_labels'].replace(classDict)

output = []
curr_word = 1
word = ''
for ind, row in pred_df.iterrows():
    if row['word'] == curr_word:
        word += row['predicted_labels']
    else:
        output.append(word)
        curr_word = row['word']
        word = row['predicted_labels']
output.append(word)
output
```




    ['Thc', 'Iuick', 'Lrcwn', 'ccx', 'JunTc', 'cxcT', 'thc', 'icIx', 'dcS']



To tackle the misclassified alphabets we shall try using a spellchecker on our predictions.


```python
# Use a spellchecker
from spellchecker import SpellChecker
english = SpellChecker()

output_corrected = []
for word in output:
    word = word.lower()
    output_corrected.append(english.correction(word))
output_corrected
```




    ['the', 'quick', 'grown', 'cox', 'junta', 'cut', 'the', 'ici', 'des']



***

<a id = '4'></a>
## 4. Deployment
For deployment we will create a web application using Flask in a form of webpage that returns the predictions after taking in an uploaded image file.

```python
from flask import Flask, flash, render_template, url_for, request, redirect
import urllib.request
import os
from werkzeug.utils import secure_filename

# import function to make prediction (Code from sections 2 and 3 of this notebook)
from recognition import make_prediction

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = 'handwriting'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
prediction = None

# Check file extensions to verify image file
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods = ['POST'])
def upload_image():    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):        
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        
        img_path = UPLOAD_FOLDER + filename
        
        prediction = make_prediction(img_path)
        
        return render_template('home.html', filename=filename, prediction = prediction)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()
```

The following shows the default home page of the web application.
![png](/images/handwriting_recognition/deployment1.png)

After selected an image file to upload and pressing "Submit" predictions will be made and returned in the following page
![png](/images/handwriting_recognition/deployment2.png)

***

<a id = '5'></a>
## 5. Final Thoughts
In this notebook I have created a basic web application that reads and predicts words in an uploaded image. Looking at the final results we observe that there are multiple mistakes in the translation of "The quick brown fox jumps over the lazy dog".

The main limitation of this model leading to these mistakes is mainly the relatively small training data used for model training (~3000 images for 52 alphabets). During training the validation accuracy of the model was observed to be approximately 80% which is relatively low for a model created to simple recognise alphabets.

By training the model on a larger dataset with more diverse examples of how each alphabet can be written the model should perform significantly better as the assumptions made for this model trivialises some of the problems faced in OCR.

Just for fun, here are some other pangrams that I've tried to run the model on
![png](/images/handwriting_recognition/deployment3.png)

