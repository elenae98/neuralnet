import torch
from torch.utils import data
import numpy as np
from PIL import Image

"""
folder = 'data/'  #change to whatever our actual 

# Create two dictionaries, one containing the IDs of images, and other containing labels of images
imageIDs = {'trainingData': ['D-1-B-N-001-001.map', 'D-1-B-N-001-002.map', 'D-1-B-N-001-003.map', 'D-1-B-N-001-004.map', 'D-1-B-N-001-005.map', 'D-1-B-N-001.map', 'D-0-A-N-001.map', 'D-0-A-N-002.map', 'D-0-A-N-003.map', 'D-0-A-N-004.map', 'D-0-A-N-005.map', 'D-0-A-Y-001-000.map', 'D-0-A-Y-001-001.map', 'D-0-A-Y-001-002.map', 'D-0-A-Y-001-003.map', 'D-0-A-Y-001-004.map', 'D-0-A-Y-001-005.map', 'D-0-A-Y-001-006.map', 'D-0-A-Y-001-007.map', 'D-0-A-Y-001-008.map', 'D-0-A-Y-001-009.map', 'D-0-A-Y-001-010.map', 'D-0-A-Y-001-011.map', 'D-0-A-Y-002-000.map', 'D-0-A-Y-002-001.map', 'D-0-A-Y-002-002.map', 'D-0-A-Y-002-003.map', 'D-0-A-Y-002-004.map', 'D-0-A-Y-002-005.map', 'D-0-A-Y-002-006.map', 'D-0-A-Y-002-007.map', 'D-0-A-Y-002-008.map', 'D-0-A-Y-002-009.map', 'D-0-A-Y-002-010.map', 'D-0-A-Y-002-011.map', 'D-0-A-Y-003-000.map', 'D-0-A-Y-003-001.map', 'D-0-A-Y-003-002.map',  'D-0-A-Y-003-003.map', 'D-0-A-Y-003-004.map', 'D-0-A-Y-003-005.map', 'D-0-A-Y-003-006.map', 'D-0-A-Y-003-007.map', 'D-0-A-Y-003-008.map', 'D-0-A-Y-003-009.map', 'D-0-A-Y-003-010.map', 'D-0-A-Y-003-011.map'], 'testData': [...]}
trainingLabels = {}

# We can fill labels dictionary using for loop and by making the names of the images something like:
# Torn_1, Torn_2, ..., notTorn_1, notTorn_2, ... (like above)
for ID in imageIDs['trainingData']:
    if ID[2] == '0':
        trainingLabels[ID] = 0 #clean
    else:
        trainingLabels[ID] = 1 #problematic
"""
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

        imgArray = genfromtxt(folder + ID, delimiter=',')
        ##include things to change array values here maybe (deal with NaN?)

        label = self.labels[ID]

        return imgArray, label