import torch
from torch.utils import data
import numpy as np
from PIL import Image

# Create two dictionaries, one containing the IDs of images, and other containing labels of images
imageIDs = {'trainingData': ['Torn_1', "Torn_2", ..., "notTorn_1", "notTorn_2", ...], 'testData' = [...]}
trainingLabels = {}

# We can fill labels dictionary using for loop and by making the names of the images something like:
# Torn_1, Torn_2, ..., notTorn_1, notTorn_2, ... (like above)
for ID in imageIDs['trainingData']:
    if "not" in ID:
        trainingLabels[ID] = 0 # not torn
    else:
        trainingLabels[ID] = 1 # torn

class TrainingSet(data.Dataset):
    def __init__(self, list_IDs, labels, folder):
        """ list_IDs = list of image IDs, this case imageIDs["trainingData"]
            labels = labels dictionary
            folder = folder that contains data, lets say "data/"
        """
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        """ Total number of testing images
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        """ Generates one sample of data
        """
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        # read file to a numpy array
        im = Image.open("data/" + ID + ".bmp") # <- not sure what type of image we will have (.jpg or something else)
        dataArray = np.array(im)
        Image.Image.close(im)
        # convert numpy array to pytorch tensor
        dataTensor = torch.from_numpy(dataArray)
        label = self.labels[ID]

        return dataTensor, label

