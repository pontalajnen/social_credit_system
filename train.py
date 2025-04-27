import torch
import time
import tqdm
import torch.nn.functional as F
import argparse

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.transforms import Compose, ToTensor, Lambda, Pad
from PIL import Image
nn = torch.nn

from models.kitchen_model import CookingNet


class CookingDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.file_names = []
        split = 'train' if train else 'test'
        with open(f'GTAV/{split}.txt', 'r') as f:
            entry = f.readline()
            while entry:
                self.file_names.append(entry.strip())
                entry = f.readline()

    def __len__(self):
        # returns the number of items in the dataset, nice and simple
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        # of the form (x, y) the values are scaled by 255 and returned as type float
        sample = (read_image(f'./noiseimages/{name}')/255., read_image(f'./images/{name}')/255.)
        return sample

    def get_plottable(self, idx):
        # Same deal as before but this time the images are permuted and not scaled
        name = self.file_names[idx]
        sample = read_image(f'./noiseimages/{name}').permute(1, 2, 0), read_image(f'./images/{name}').permute(1, 2, 0)
        return sample


class Trainer():
    def __init__(self, model, batch_size,  opt, lr):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model = model(True)

        self.model = model.to(self.device, dtype=torch.float)

        self.train_data = DataLoader(CookingDataset(True), batch_size=batch_size, shuffle=True)
        self.test_data = DataLoader(CookingDataset(False), batch_size=batch_size, shuffle=True)

        self.optimizer = opt(model.parameters(), lr)

    def train(self, epochs):
        loss_history = torch.zeros((5))
        lh = []
        start = time.time()
        for epoch in range(epochs):
            print('Epoch: ', epoch+1)
            progress_bar = tqdm(self.train_data)
            for e, (x, y) in enumerate(progress_bar, 1):
                x = x.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.float)
                pred = self.model(x)
                loss = F.mse_loss(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                n = min(e, 5)
                # .item() takes the single tensor value and makes it a python native element
                loss_history[e % 5] = loss.item()
                lh.append(loss.item())
                # this line will print the rolling loss average on the progress bar
                progress_bar.set_postfix(Loss=(loss_history.sum()/n).item())
        print('Total Training Time: ', time.time() - start)
        return lh

    @torch.inference_mode()
    def test(self):
        # This function calculates the mean average distance of the pixels
        # and subtracts it from 1 to get an accuracy
        def noiseAccuracy(predicted, target):
            size = target.flatten().shape[0]
            true = (predicted - target).abs().mean()
            return 1-true
        total_accuracy = 0
        start = time.time()
        progress_bar = tqdm(self.test_data)
        total_steps = 0

        for e, (x, y) in enumerate(pbar, 1):
            x = x.to(self.device, dtype=torch.float)
            y = y.to(self.device, dtype=torch.float)
            pred = self.model(x)

            # This is the accuracy of the model
            accuracy = noiseAccuracy(pred, y).item()
            total_accuracy += accuracy
            total_steps = e

            progress_bar.set_postfix(Accuracy=total_accuracy/total_steps)

        print('Noise Accuracy: ' + str((total_accuracy/(total_steps-1))))
        print('Total Eval Time: ', time.time() - start)


def train(test):

    if test:
        model = torch.load('model.pth')
    model = CookingNet()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument("-t", "--test", action="store_true", help="Test the model after training")
    args = parser.parse_args()
    train(args.test)
