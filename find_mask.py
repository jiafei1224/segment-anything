import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

def count_items_in_folder(folder_path):
    try:
        # List all files and folders in the given directory
        items = os.listdir(folder_path)
        
        # Count the number of items
        num_items = len(items)
        
        return num_items
    except FileNotFoundError:
        print(f"The folder {folder_path} was not found.")
        return 0
    except PermissionError:
        print(f"Do not have permission to access the folder {folder_path}.")
        return 0


folder_path = '/home/duanj1/CameraCalibration/Detectron2DeepSortPlus/frames'
num_items = count_items_in_folder(folder_path)

my_list = list(range(1, num_items + 1))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# Assuming you have a function get_coordinates_by_id
def get_coordinates_by_id(dataframe, id_value):
    rows = dataframe.query(f'ID == "{id_value}"')
    coordinates_list = []
    for index, row in rows.iterrows():
        x = float(row['X'])
        y = float(row['Y'])
        coordinates_list.append([x, y])
    return coordinates_list

def generate_black_white_mask(mask):
    binary_mask = (mask > 0).astype(np.uint8)
    black_white_mask = binary_mask * 255
    return black_white_mask

# Create directory for saving masks if it doesn't exist
save_dir = "saved_masks"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Read the text file and convert it into a Pandas DataFrame
columns = ['ID', 'X', 'Y']
data = []
with open('/home/duanj1/CameraCalibration/Detectron2DeepSortPlus/output_pose.txt', 'r') as file:
    for line in file:
        row = line.strip().split(',')
        data.append(row)
df = pd.DataFrame(data, columns=columns)

# Find all unique IDs in the DataFrame
unique_ids = [int(x) for x in df['ID'].unique()]

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "/home/duanj1/CameraCalibration/segment_anything/sam_vit_h_4b8939 (3).pth"
model_type = "vit_h"

device = "cuda"

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

# predictor = SamPredictor(sam)


# predictor.set_image(image)
previous_valid_coord = None

# for i in my_list:
#     if i not in unique_ids:
#         # Create a black mask of the same size as the existing image
#         image_path = f'/home/duanj1/CameraCalibration/Detectron2DeepSortPlus/frames/{i}.png'
#         image = cv2.imread(image_path)
#         if image is not None:
#             black_mask = np.zeros_like(image[:,:,0])
            
#             # Save the black mask
#             save_path = os.path.join(save_dir, f"{i}.png")
#             cv2.imwrite(save_path, black_mask)
#             print(f"Saved black mask for ID {i} at {save_path}")

#     else:
my_list=[]
for i in unique_ids:
    coord_list = get_coordinates_by_id(df, i)
    first = get_coordinates_by_id(df, 58)

    if len(coord_list) == 0:
        continue

    CoordiXY = coord_list[0]
    print(CoordiXY)

    # Read corresponding image
    image_path = f'/home/duanj1/CameraCalibration/Detectron2DeepSortPlus/frames/{i}.png'
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)


#     predictor.set_image(image)

    # Set image for SAM predictor
    predictor.set_image(image)

    # Make prediction to generate mask
    input_point = np.array([CoordiXY])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Use mask with the highest score
    highest_score_index = np.argmax(scores)
    highest_score_mask = masks[highest_score_index]

    bw_mask = generate_black_white_mask(highest_score_mask)

    # Check the white pixel ratio
#         total_pixels = bw_mask.size
    white_pixels = np.sum(bw_mask == 255)
    white_ratio = white_pixels / 2073600
    print(white_ratio)

    exhausted_list = False  # Flag to check if we have looped through all elements in my_list

    while white_ratio > 0.3 and not exhausted_list:  # Modified while loop condition

        if previous_valid_coord is not None:

            for index, x in enumerate(my_list):
                # Reprocess using the previous valid coordinates
                print(i)
                input_point = np.array([x])

                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )

                highest_score_index = np.argmax(scores)
                highest_score_mask = masks[highest_score_index]
                bw_mask = generate_black_white_mask(highest_score_mask)

                white_pixels = np.sum(bw_mask == 255)
                white_ratio = white_pixels / 2073600
                print(white_ratio)

                if white_ratio < 0.3:
                    break  # Stop the for-loop since the condition is met

            if index == len(my_list) - 1:
                exhausted_list = True
            
                
                # Mark that we have looped through all elements in my_list

        else:
            print(f"Warning: No previous valid coordinates for ID {i}.")
            exhausted_list = True  # Exit the loop since we have no previous valid coordinates to proceed with
#     if white_ratio >0.3 and exhausted_list:
#         bw_mask = np.zeros((height, width, 3), dtype=np.uint8)
#         print("Generated Black")
    if white_ratio > 0.3:
        height, width = bw_mask.shape  # Get dimensions from the last bw_mask
        bw_mask = np.zeros((height, width), dtype=np.uint8)  # Create an all-black mask
        print("Generated Black")
        save_path = os.path.join(save_dir, f"{i}.png")
        cv2.imwrite(save_path, bw_mask)
    
    if white_ratio <=0.3:
        # Save this coordinate as the previous successful coordinate for the next iteration
        previous_valid_coord = CoordiXY
        my_list.append(previous_valid_coord)
        save_path = os.path.join(save_dir, f"{i}.png")
        cv2.imwrite(save_path, bw_mask)

        print(f"Saved mask for ID {i} at {save_path}")

    # Save the mask
    


# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()