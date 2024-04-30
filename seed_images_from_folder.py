import cv2
import os
from seed_segmenter import separate_seeds
import database as db


# For naming and database
seed_type = "kaura"
use_type = "test"
run = 0

# Save destination
folder_path = f"data/raw/{use_type}/{seed_type}"
save_dest = f"data/images/{use_type}/{seed_type}"

# Get files
file_list = os.listdir(folder_path)

# For image cropping
borders = [50, 1350]


i = 0
for file in file_list:
    path = os.path.join(folder_path, file)
    # Use only files ending with 
    if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image = cv2.imread(path)
        # Get segmented images
        images = separate_seeds(image, crop_left_right=borders,label_dilation=2)
        if len(images) > 0:
            j=0
            for im in images:
                filename = f"image{run}_{i}_{j}.png"
                save_path = f"{save_dest}/{filename}"
                # Add info to database
                db.insert(seed_type, save_path, use_type)
                # Save segmented image
                cv2.imwrite(save_path, im)
                j+=1
        i+=1
    # Tell progress
    print(f"Completion: {i} / {len(file_list)}")







    
    
