"""
Choose the file and apply the selected model to obtain a prediction. The model is saved in random_forest_model.sav.
"""

import os
import sklearn as skl
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import filedialog



model = pickle.load(open("random_forest_model.sav", 'rb'))

directory = "holdout"

failed_files = []

def load_image(file, i):
    try:
        if not file.lower().endswith(".png"):
            print(f"Skipping non-image file {file}")
            return None
        print(f"Loading {file}" if i == 0 else f"Loading {file}")
        img = imread(file, as_gray=False)
        # if image is grayscale, convert to RGB
        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
        resized_img = resize(img, (150, 150, 3)).flatten()
        label = i
        return resized_img, label
    except EOFError:
        print(f"EOFError reading {file}" if i == 0 else f"EOFError reading {file}")
        failed_files.append(file)
        return None
    except OSError:
        print(f"OSError reading {file}" if i == 0 else f"OSError reading {file}")
        failed_files.append(file)
        return None
    except ValueError:
        print(f"ValueError reading {file}" if i == 0 else f"ValueError reading {file}")
        failed_files.append(file)
        return None
    
def main():


    data = []
    labels = []

    for i in range(4):
        path = os.path.join(directory, "kaura" if i == 0 else "ohra" if i == 1 else "ruis" if i == 2 else "vehna")
        files = [os.path.join(path, file) for file in os.listdir(path)]
        for file in files:
            result = load_image(file, i)
            if result is not None:
                img, label = result
                data.append(img)
                labels.append(label)
                print(f"Loaded {file}" if i == 0 else f"Loaded {file}")

    data = np.array(data)
    labels = np.array(labels)

    x_test, y_test = data, labels

    x_test = x_test.reshape(x_test.shape[0], -1)

    print("Accuracy: " + str(model.score(x_test, y_test)))

    # confusion matrix

    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    # Convert to percentages
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    print(cm_perc)



    while True:

        try:

            # open a file browser to select a file from your computer

            root = tk.Tk()
            root.withdraw()

            file_path = filedialog.askopenfilename()

            # read the image

            img = load_image(file_path, 0)[0]

            # resize the image

            # predict the digit

            prediction = model.predict([img])

            if prediction[0] == 0:
                print("Prediction: kaura")
            elif prediction[0] == 1:
                print("Prediction: ohra")
            elif prediction[0] == 2:
                print("Prediction: ruis")
            elif prediction[0] == 3:
                print("Prediction: vehnä")


        except IOError:

            break

if __name__ == "__main__":
    main()



