'''

This script contains functions to pre-process the images by adding biases (i.e. blur & color temperature perturbation)

- convert_temp: perturb the color temperature bias by scaling different channels

- create_clear_annotation: get the annotation files from an existing dataset

- perturb_color_multi_bias: perturb the dataset by color temperature in multi-bias

- perturb_color_specific_bias: perturb the dataset by color temperature in a specific bias

- perturb_blur_multi_bias: perturb the dataset by blur in multi-bias

- perturb_blur_specific_bias: perturb the dataset by blur in a specific bias

'''

from PIL import Image
import os 
import cv2
import random
import argparse
import pandas as pd

# Color temperature mapping table for data bias.
kelvin_table = {
    1000: (255, 56, 0),
    1100: (255, 71, 0),
    1200: (255, 83, 0),
    1300: (255, 93, 0),
    1400: (255, 101, 0),
    1500: (255, 109, 0),
    1600: (255, 115, 0),
    1700: (255, 121, 0),
    1800: (255, 126, 0),
    1900: (255, 131, 0),
    2000: (255, 137, 18),
    2100: (255, 142, 33),
    2200: (255, 147, 44),
    2300: (255, 152, 54),
    2400: (255, 157, 63),
    2500: (255, 161, 72),
    2600: (255, 165, 79),
    2700: (255, 169, 87),
    2800: (255, 173, 94),
    2900: (255, 177, 101),
    3000: (255, 180, 107),
    3100: (255, 184, 114),
    3200: (255, 187, 120),
    3300: (255, 190, 126),
    3400: (255, 193, 132),
    3500: (255, 196, 137),
    3600: (255, 199, 143),
    3700: (255, 201, 148),
    3800: (255, 204, 153),
    3900: (255, 206, 159),
    4000: (255, 209, 163),
    4100: (255, 211, 168),
    4200: (255, 213, 173),
    4300: (255, 215, 177),
    4400: (255, 217, 182),
    4500: (255, 219, 186),
    4600: (255, 221, 190),
    4700: (255, 223, 194),
    4800: (255, 225, 198),
    4900: (255, 227, 202),
    5000: (255, 228, 206),
    5100: (255, 230, 210),
    5200: (255, 232, 213),
    5300: (255, 233, 217),
    5400: (255, 235, 220),
    5500: (255, 236, 224),
    5600: (255, 238, 227),
    5700: (255, 239, 230),
    5800: (255, 240, 233),
    5900: (255, 242, 236),
    6000: (255, 243, 239),
    6100: (255, 244, 242),
    6200: (255, 245, 245),
    6300: (255, 246, 247),
    6400: (255, 248, 251),
    6500: (255, 249, 253),
    6600: (254, 249, 255),
    6700: (252, 247, 255),
    6800: (249, 246, 255),
    6900: (247, 245, 255),
    7000: (245, 243, 255),
    7100: (243, 242, 255),
    7200: (240, 241, 255),
    7300: (239, 240, 255),
    7400: (237, 239, 255),
    7500: (235, 238, 255),
    7600: (233, 237, 255),
    7700: (231, 236, 255),
    7800: (230, 235, 255),
    7900: (228, 234, 255),
    8000: (227, 233, 255),
    8100: (225, 232, 255),
    8200: (224, 231, 255),
    8300: (222, 230, 255),
    8400: (221, 230, 255),
    8500: (220, 229, 255),
    8600: (218, 229, 255),
    8700: (217, 227, 255),
    8800: (216, 227, 255),
    8900: (215, 226, 255),
    9000: (214, 225, 255),
    9100: (212, 225, 255),
    9200: (211, 224, 255),
    9300: (210, 223, 255),
    9400: (209, 223, 255),
    9500: (208, 222, 255),
    9600: (207, 221, 255),
    9700: (207, 221, 255),
    9800: (206, 220, 255),
    9900: (205, 220, 255),
    10000: (207, 218, 255),
    10100: (207, 218, 255),
    10200: (206, 217, 255),
    10300: (205, 217, 255),
    10400: (204, 216, 255),
    10500: (204, 216, 255),
    10600: (203, 215, 255),
    10700: (202, 215, 255),
    10800: (202, 214, 255),
    10900: (201, 214, 255),
    11000: (200, 213, 255),
    11100: (200, 213, 255),
    11200: (199, 212, 255),
    11300: (198, 212, 255),
    11400: (198, 212, 255),
    11500: (197, 211, 255),
    11600: (197, 211, 255),
    11700: (197, 210, 255),
    11800: (196, 210, 255),
    11900: (195, 210, 255),
    12000: (195, 209, 255)}

blur_norm_table_imagenette = {
    '8':29, '16':53, '24':79, '32':103
}

def convert_temp(image, temp):
    r, g, b = kelvin_table[temp]
    color_matrix = ( r / 255.0, 0.0, 0.0, 0.0,
                    0.0, g / 255.0, 0.0, 0.0,
                    0.0, 0.0, b / 255.0, 0.0 )
    return image.convert('RGB', color_matrix)

def create_clear_annotation(split_type='train'):
    g = os.walk("./datasets/imagenette/images/nobias/{}".format(split_type))  
    filename = list()
    label = list()
    for path, dir_list, file_list in g:
        category = path.split("/")[-1]
        for file_name in file_list:
            filename.append(os.path.join(path, file_name))
            label.append(category)

    df = pd.DataFrame({'filename':filename, 'label':label})
    df['gaussian_kernel'] = [0] * len(df)
    df['color_temp'] = [6600] * len(df)
    
    # Dropping images with invalid format
    invalid_index = list()
    for i in df.index:
        source_img_path = df['filename'][i]
        source_img = Image.open(source_img_path)
        if source_img.mode != 'RGB': 
            invalid_index.append(i)
    df.drop(index=invalid_index, inplace=True)
    df.to_csv("./datasets/imagenette/nobias_{}.csv".format(split_type), index=False)
    return 0

def perturb_color_multi_bias(split_type='train'):
    # Create the annotation file for biased data.
    bias_data_dir = "ct_multibias"
    df = pd.read_csv("./datasets/imagenette/nobias_{}.csv".format(split_type), dtype=str)
    df['filename'] = [imgpath.replace("nobias", bias_data_dir) for imgpath in df["filename"]]
    # Determine the color temperature bias level for images randomly in a range
    # The unit is Kelvin // 100.
    df['color_temp'] = [str(10+random.randint(0, 110)*1) for imgpath in df["filename"]]
    df.to_csv("./datasets/imagenette/{}_{}.csv".format(bias_data_dir, split_type), index=False)
    
    # Preturb color temperature bias level for images based on the assignment.
    for i in df.index:
        target_img_path = df['filename'][i]
        source_img_path = target_img_path.replace(bias_data_dir, "nobias")
        folder_path = '/'.join(target_img_path.split('/')[:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        source_img = Image.open(source_img_path)
        target_img = convert_temp(source_img, int(df['color_temp'][i])*100)
        target_img.save(target_img_path)
    return 0

def perturb_color_specific_bias(split_type='train', kelvin=0):
    # Create the annotation file for biased data.
    bias_data_dir = "ct_"+str(kelvin)
    abs_kelvin = int(kelvin)+6600
    df= pd.read_csv("./datasets/imagenette/nobias_{}.csv".format(split_type), dtype=str)
    df['filename'] = [imgpath.replace("nobias", bias_data_dir) for imgpath in df["filename"]]
    df['color_temp'] = [abs_kelvin//100] * len(df)
    df.to_csv("./datasets/imagenette/{}_{}.csv".format(bias_data_dir, split_type), index=False)

    # Preturb color temperature bias to a specific level
    for i in df.index:
        target_img_path = df['filename'][i]
        source_img_path = target_img_path.replace(bias_data_dir, "nobias")
        folder_path = '/'.join(target_img_path.split('/')[:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        source_img = Image.open(source_img_path)
        target_img = convert_temp(source_img, abs_kelvin)
        target_img.save(target_img_path)
    return

def perturb_blur_multi_bias(split_type='train'):
    # Create the annotation file for biased data.
    bias_data_dir = "blur_multibias"
    df = pd.read_csv("./datasets/imagenette/nobias_{}.csv".format(split_type), dtype=str)
    df['filename'] = [imgpath.replace("nobias", bias_data_dir) for imgpath in df["filename"]]
    # Determine the blur bias level for images randomly in a range
    df['gaussian_kernel'] = [str(1+random.randint(0, 80)*2) for imgpath in df["filename"]]
    df.to_csv("./datasets/imagenette/{}_{}.csv".format(bias_data_dir, split_type), index=False)
    
    # Preturb blur bias to a specific level
    for i in df.index:
        target_img_path = df['filename'][i]
        source_img_path = target_img_path.replace(bias_data_dir, "nobias")
        folder_path = '/'.join(target_img_path.split('/')[:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        source_img = cv2.imread(source_img_path)
        filter_sz = int(df['gaussian_kernel'][i])
        target_img = cv2.GaussianBlur(source_img, (filter_sz, filter_sz), 0) 
        cv2.imwrite(target_img_path, target_img)
    return

def perturb_blur_specific_bias(split_type='train', sigma=0):
    # Create the annotation file for biased data.
    filter_sz=blur_norm_table_imagenette[sigma]
    bias_data_dir = "blur_{}".format(sigma)
    df = pd.read_csv("./datasets/imagenette/nobias_{}.csv".format(split_type), dtype=str)
    df['filename'] = [imgpath.replace("nobias", bias_data_dir) for imgpath in df["filename"]]
    df['gaussian_kernel'] = [filter_sz] * len(df)
    df.to_csv("./datasets/imagenette/{}_{}.csv".format(bias_data_dir, split_type), index=False)
    
    # Preturb blur bias to a specific level
    for i in df.index:
        target_img_path = df['filename'][i]
        source_img_path = target_img_path.replace(bias_data_dir, "nobias")
        folder_path = '/'.join(target_img_path.split('/')[:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        source_img = cv2.imread(source_img_path)
        filter_sz = int(filter_sz)
        target_img = cv2.GaussianBlur(source_img, (filter_sz, filter_sz), 0) 
        cv2.imwrite(target_img_path, target_img)
    return 
