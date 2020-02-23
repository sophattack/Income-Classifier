import torch.utils.data as data
import torch


class AdultDataset(data.Dataset):

    def __init__(self, X, y, transforms=None):
        ######
        # 4.1 YOUR CODE HERE
        self.X = X
        self.y = y
        self.transforms = transforms
        ######

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        ######

        # 4.1 YOUR CODE HERE

        X = self.X[index]
        y = self.y[index]

        if self.transforms:
            X = self.transforms(self.X)

        return X, y
        ######
