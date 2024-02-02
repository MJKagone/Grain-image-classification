"""
Loads a dataset of images of 4 kinds of different seeds represented in jpg format, trains a random forest classifier and a kNN classifier on the data, and tests the models.

The data is kept in a directory structure as follows:
train >
    kaura
    ohra
    ruis
    vehna
"""


import os
import sklearn as skl
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
import pickle
from skimage.transform import resize



directory = "train"

data = []
labels = []

failed_files = []


def prepare_data():
  
    for i in range(4):
        path = os.path.join(directory, "kaura" if i == 0 else "ohra" if i == 1 else "ruis" if i == 2 else "vehna")
        files = [os.path.join(path, file) for file in os.listdir(path)]
        for file in files:
            result = load_image(file, i)
            if result is not None:
                features, label = result
                data.append(features)
                labels.append(label)
                print(f"Loaded {file}" if i == 0 else f"Loaded {file}")

    data_np = np.array(data)
    labels_np = np.array(labels)
    if not failed_files:
        print("All files loaded successfully")
    else:
        print(f"Failed to load files {failed_files}")

    # save the data in a file

    print("Saving data...")
    np.save("data.npy", data_np)
    np.save("labels.npy", labels_np)

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
        resized_img = resize(img, (150, 150, 3))
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
    # except ValueError:
    #     print(f"ValueError reading {file}" if i == 0 else f"ValueError reading {file}")
    #     failed_files.append(file)
    #     return None

        

def extract_features():
    
    images = np.load("data.npy")
    images_gray = np.zeros((images.shape[0], images.shape[1], images.shape[2]))

    # save the features in a file

    np.save("features.npy", images_gray)





def split_data():
        
    x_train, x_validation, y_train, y_validation = train_test_split(np.load("features.npy"), np.load("labels.npy"), test_size=0.2, shuffle=True, stratify=np.load("labels.npy"))

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_validation = x_validation.reshape(x_validation.shape[0], -1)

    return x_train, x_validation, y_train, y_validation

# train random forest model

# use RandomizedSearchCV to find the best hyperparameters
    
def train_models(x_train, y_train, x_validation, y_validation):

    print("Training random forest model...")
    random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
    random_forest_model.fit(x_train, y_train)
    pickle.dump(random_forest_model, open("random_forest_model.sav", 'wb'))

    # print("Training neural network model...")
    # neural_network_model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
    # neural_network_model.fit(x_train, y_train)
    # pickle.dump(neural_network_model, open("neural_network_model.sav", 'wb'))

    # print("Training kNN model...")
    # knn_3_model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    # knn_3_model.fit(x_train, y_train)
    # pickle.dump(knn_3_model, open("knn_model_3.sav", 'wb'))


"""
# load actual test set

directory = "MNIST Dataset JPG format"

data = []
labels = []

for i in range(10):
    path = os.path.join(directory, "MNIST - JPG - Testing", str(i))
    for file in os.listdir(path):
        img = imread(os.path.join(path, file), as_gray=True).flatten()
        data.append(img)
        labels.append(i)

data = np.array(data)
labels = np.array(labels)

x_validation = data
y_validation = labels
"""

def evaluate_models(x_validation, y_validation):
    random_forest_model = pickle.load(open("random_forest_model.sav", 'rb'))
    print("Random forest accuracy: ", random_forest_model.score(x_validation, y_validation))



def main():

    # prepare_data()
    extract_features()
    x_train, x_validation, y_train, y_validation = split_data()
    train_models(x_train, y_train, x_validation, y_validation)
    evaluate_models(x_validation, y_validation)

if __name__ == "__main__":
    main()

