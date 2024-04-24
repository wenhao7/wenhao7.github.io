---
layout: post
title:  "Python Product Image Editing Tool"
date:   2024-04-24 16:11:09 +0800
category: [machine_learning, deep_learning, misc]
tag: [numpy, pytorch, transfer_learning, classification, object_detection, computer_vision, automation, featured]
summary: "In a [previous notebook](https://wenhao7.github.io/machine_learning/deep_learning/misc/2024/04/22/object_detection_poc.html) we have taken a look at how e-commerce product photos can be automatically processed into various resolutions for different purposes. Now we are going to explore building a simple application using Python to utilize the ideas that we have implemented. A copy of the code to this application can be found on [its repository](https://github.com/wenhao7/PyPIET/tree/main)."
image: /images/banners/pypiet.png
---

## Contents
1. [Introduction](#1)
2. [Implemented Approach](#2)
3. [Building The Interface](#3)
4. [Conclusion](#4)

<a id='1'></a>
## 1. Introduction
In a [previous notebook](https://wenhao7.github.io/machine_learning/deep_learning/misc/2024/04/22/object_detection_poc.html) we have taken a look at how e-commerce product photos can be automatically processed into various resolutions for different purposes. Now we are going to explore building a simple application using Python to utilize the ideas that we have implemented. A copy of the code to this application can be found on [its repository](https://github.com/wenhao7/PyPIET/tree/main).

<a id='2'></a>
## 2. Implemented Approach
In this section we look at results from the previous notebook's implementation. Using the same stock image as our test subject we requested for two images at a resolution of 960x960, one to showcase the sweater and another to showcase the trousers.


```python
import numpy as np
import cv2
from ultralytics import YOLO
from IPython.display import Image, display
import PIL
from PIL import ImageCms
import os
import io
import matplotlib.pyplot as plt

from resize_app_helper import *
```


```python
test_img_path = 'images/pexels-amina-filkins-5560019.jpg'
model = YOLO("models/best.pt")
person_model = YOLO("models/yolov8l.pt")

top_bottom_classes = list(range(9))
res = detect(model, test_img_path, top_bottom_classes)
person_res = detect(person_model, test_img_path, 0)
cropped_imgs, paddings, product_classes = crop_image(test_img_path, res, person_res=person_res)

target_res = (960,960)
test_img = PIL.Image.open(test_img_path)
print('Original Image')
display(test_img.resize((test_img.size[0]//4, test_img.size[1]//4), PIL.Image.LANCZOS))
for i in range(len(cropped_imgs)):
    resized_img_arr = resize_pad(cropped_imgs[i], padding=paddings[i], target_res=target_res, product_class=product_classes[i])
    resized_img = PIL.Image.fromarray(resized_img_arr)
    print(f"Displaying Edited Image for {product_classes[i]} at {resized_img.size} resolution")
    display(resized_img)
```

    
    image 1/1 C:\Users\wenhao\Desktop\Job\ML\yolo_objectdetection\images\pexels-amina-filkins-5560019.jpg: 640x448 1 long_sleeve_top, 1 trousers_bottom, 115.5ms
    Speed: 5.5ms preprocess, 115.5ms inference, 1341.0ms postprocess per image at shape (1, 3, 640, 448)
    
    image 1/1 C:\Users\wenhao\Desktop\Job\ML\yolo_objectdetection\images\pexels-amina-filkins-5560019.jpg: 640x448 1 person, 104.0ms
    Speed: 2.5ms preprocess, 104.0ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 448)
    Original Image
    


    
![png](/images/pypiet/output_2_1.png)
    


    Displaying Edited Image for long_sleeve_top at (960, 960) resolution
    


    
![png](/images/pypiet/output_2_3.png)
    


    Displaying Edited Image for trousers_bottom at (960, 960) resolution
    


    
![png](/images/pypiet/output_2_5.png)
    


From the results we see that both images are nicely scaled and centred with the subject position correctly in the middle with even spacing surrounding it. The only visible is some artifacts from where the background was padded and blended, this limitation was discussed in the previous notebook and can likely be resolved by finetuning how we are blending the original and extend background. 

Nevertheless there is still value to the images created by this tool as the subject should never be affected by this artifacts and the results still look really good compared to the labour required to produce it.

<a id='3'></a>
## 3. Building the Interface
In this section we will look at how a User Interface is built so that the tool can be quickly utilized by anyone. Package `tkinter` that comes prepackaged with Python will be used for this purpose, the full code can be found [here](https://github.com/wenhao7/PyPIET/blob/main/main.py).

Below is a screenshot of our end product.

![png](/images/pypiet/ui_ss.PNG)

Within our class for the tkinter object we have our class attributes to store the folder path of our input images and a dictionary containing our desired output paths and resolutions.

    class ResizeApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Resizing App")
            self.folder_path = "select folder path using the button on the left"
            self.reso_dict = {}
            
The buttons within the toolbar at the top are created by creating a Frame widget to contain and organize the buttons and labels. Positioning is managed by the splitting the frame into a grid of rows and columns. The commands are user defined functions, `select_folder` allows user to choose a folder path through their system's file explorer, `open_file` lets user open and edit a text file within the UI to specify the desired output path and resolutions, `save_file` will save the text file to disk. `run` executes the workflow to convert the input images.

    self.top_frame = tk.Frame(root, height=100)
    self.top_frame.pack(side=tk.TOP)
    self.top_frame.columnconfigure(1, weight=4, minsize=300)

    self.filedir_button = tk.Button(self.top_frame, text="Folder..", command=self.select_folder)
    self.filedir_label = tk.Label(self.top_frame, bg='white')
    self.open_button = tk.Button(self.top_frame, text='Edit Resolutions', command=self.open_file)
    self.save_button = tk.Button(self.top_frame, text='Save Resolutions', command=self.save_file)

    self.filedir_button.grid(row=0, column=0, pady=3)
    self.filedir_label.grid(row=0, column=1,columnspan=3, padx=5, pady=3)
    self.open_button.grid(row=0, column=4, sticky=tk.NE, pady=3)
    self.save_button.grid(row=0, column=5, sticky=tk.NE, pady=3)

    self.run_button = tk.Button(self.top_frame, text='Run', command=self.run)
    self.run_button.grid(row=0, column=6)
    
Below we see a sample of what the resolutions text file may look like

![png](/images/pypiet/ui_reso.PNG)

In the bottom part of the UI we have another Frame widget, containing a Text Widget that can display text files (Readme doc and user's resolution file)
User defined function `start_up` reads the tool's readme file and the resolution text file everytime the tool is launched

    self.bottom_frame = tk.Frame(root)
    self.bottom_frame.pack(side=tk.BOTTOM)
    self.text_widget = tk.Text(self.bottom_frame, wrap=tk.WORD)
    self.text_widget.grid(row=0)  

    self.start_up()
    
The below `run` function is the main function applying the image editing process to our input images.<br>
`get_filepaths` taking in the chosen folder path, returns complete file paths for all images within the chosen input folder.<br>
`process_images_path` taking in the list of filepaths found, returns relative folder path of images.<br>
`process_image` will take a file, editing, creating, and saving an image for each target resolution provided.<br>
The tool will process the entire input folder sequentially creating a final image for the entire model/subject, and a final image zooming into detected products while displaying progress in Python's output.

    def run(self):
        model = YOLO('models/best.pt')
        person_model = YOLO('models/yolov8l.pt')
        top_bottom_classes = list(range(13))
        
        file_paths = get_filepaths(self.folder_path + '/')
        folder_name = process_images_path(file_paths)
        start_time = datetime.now()
        print(f'Starting Time: {str(start_time)}')
        for i in range(len(file_paths)):
            s = f' =============== Progress : {i+1}/{len(file_paths)} files completed ==============='
            file = file_paths[i]
            process_image(file, self.reso_dict, folder_name)
            print(s)
            
        print(f" ***************** {i+1} / {i+1} files completed ***************** ")
        end_time = datetime.now()
        print(f'Ending Time: {str(end_time)}')
        time_delta = str(timedelta(seconds=(end_time-start_time).total_seconds()))
        print(f'Time Taken : {str(time_delta)}')
        
        
<a id='4'></a>
## 4. Conclusion
In this notebook we have explored how we can quickly translate our [proof of concept](https://wenhao7.github.io/machine_learning/deep_learning/misc/2024/04/22/object_detection_poc.html) into a UI that is easily utilized by anyone with minimal technical knowledge. For users who are editing a large number of product photos for e-commerce or marketing purpose this tool may prove to be extremely useful, automatically completing what is a very tedious and time consuming task.
