import os
import time
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import albumentations as A
from scipy.ndimage import binary_dilation
import segmentation_models_pytorch as smp
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils_ import MriDataset, EarlyStopping

class Main():

    def __init__(self,df, epochs) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = df
        self.epochs = epochs
        self.model = self.create_model()
        self.imputer = SimpleImputer(strategy="most_frequent")
        self.batch_size = 16

    def get_file_row(self, path):
        """Extracts patient ID, image filename, and mask filename from the given path."""
        path_no_ext, ext = os.path.splitext(path)
        filename = os.path.basename(path)
        
        # Extract patient ID from the first three segments of the filename
        patient_id = '_'.join(filename.split('_')[:3])
        
        return [patient_id, path, f'{path_no_ext}_mask{ext}']

    def iou_pytorch(self, predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
        """Calculates Intersection over Union for a tensor of predictions."""
        predictions = torch.where(predictions > 0.5, 1, 0)
        labels = labels.byte()

        intersection = (predictions & labels).float().sum((1, 2))
        union = (predictions | labels).float().sum((1, 2))

        iou = (intersection + e) / (union + e)
        return iou

    def dice_pytorch(self, predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
        """Calculates Dice coefficient for a tensor of predictions."""
        predictions = torch.where(predictions > 0.5, 1, 0)
        labels = labels.byte()

        intersection = (predictions & labels).float().sum((1, 2))
        dice_coefficient = ((2 * intersection) + e) / (
            predictions.float().sum((1, 2)) + labels.float().sum((1, 2)) + e
        )
        return dice_coefficient

    def BCE_dice(self, output, target, alpha=0.01):
        """Combines binary cross-entropy and Dice loss with an optional weight."""
        bce = torch.nn.functional.binary_cross_entropy(output, target)
        soft_dice = 1 - self.dice_pytorch(output, target).mean()
        return bce + alpha * soft_dice

    def training_loop(self, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler):
        """Training loop for a PyTorch model."""
        history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
        early_stopping = EarlyStopping(patience=7)

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            running_loss = 0
            self.model.train()
            for i, data in enumerate(tqdm(train_loader)):
                img, mask = data
                img, mask = img.to(self.device), mask.to(self.device)
                predictions = self.model(img)
                predictions = predictions.squeeze(1)
                loss = loss_fn(predictions, mask)
                running_loss += loss.item() * img.size(0)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            self.model.eval()
            with torch.no_grad():
                running_IoU = 0
                running_dice = 0
                running_valid_loss = 0
                for i, data in enumerate(valid_loader):
                    img, mask = data
                    img, mask = img.to(self.device), mask.to(self.device)
                    predictions = self.model(img)
                    predictions = predictions.squeeze(1)
                    running_dice += self.dice_pytorch(predictions, mask).sum().item()
                    running_IoU += self.iou_pytorch(predictions, mask).sum().item()
                    loss = loss_fn(predictions, mask)
                    running_valid_loss += loss.item() * img.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            val_loss = running_valid_loss / len(valid_loader.dataset)
            val_dice = running_dice / len(valid_loader.dataset)
            val_IoU = running_IoU / len(valid_loader.dataset)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_IoU'].append(val_IoU)
            history['val_dice'].append(val_dice)
            
            elapsed_time = time.time() - start_time
            print(f'Epoch: {epoch}/{self.epochs} | Training loss: {train_loss:.4f} | Validation loss: {val_loss:.4f} | '
                f'Validation Mean IoU: {val_IoU:.4f} | Validation Dice coefficient: {val_dice:.4f} | '
                f'Time: {elapsed_time:.2f}s')

            lr_scheduler.step(val_loss)
            if early_stopping(val_loss, self.model):
                early_stopping.load_weights(self.model)
                break

        self.model.eval()
        return history

    def show_evaluation(self, history):
        plt.figure(figsize=(7, 7))
        plt.plot(history['train_loss'], label='Training loss')
        plt.plot(history['val_loss'], label='Validation loss')
        plt.ylim(0, 0.01)
        plt.legend()
        plt.show()

        plt.figure(figsize=(7, 7))
        plt.plot(history['val_IoU'], label='Validation mean Jaccard index')
        plt.plot(history['val_dice'], label='Validation Dice coefficient')
        plt.legend()
        plt.show()

    def show_sample(self, train_dataset):
        n_examples=4
        fig, axs = plt.subplots(n_examples, 3, figsize=(5, n_examples*2), constrained_layout=True)
        i = 0
        for ax in axs:
            while True:
                image, mask = train_dataset.__getitem__(i, raw=True)
                i += 1
                if np.any(mask): 
                    ax[0].set_title("MRI images")
                    ax[0].imshow(image)
                    ax[1].set_title("Highlited abnormality")
                    ax[1].imshow(image)
                    ax[1].imshow(mask, alpha=0.2)
                    ax[2].imshow(mask)
                    ax[2].set_title("Abnormality mask")
                    break
        plt.show()

    def show_testing(self, test_loader):
        width = 2
        columns = 5
        n_examples = columns * width

        fig, axs = plt.subplots(columns, width, figsize=(7*width , 7*columns), constrained_layout=True)
        red_patch = mpatches.Patch(color='red', label='The red data')
        fig.legend(loc='upper right',handles=[
        mpatches.Patch(color='red', label='Ground truth'),
        mpatches.Patch(color='green', label='Predicted abnormality')])
        i = 0
        with torch.no_grad():
            for data in test_loader:
                image, mask = data
                mask = mask[0]
                if not mask.byte().any():
                    continue
                image = image.to(self.device)
                prediction = self.model(image).to('cpu')[0][0]
                prediction = torch.where(prediction > 0.5, 1, 0)
                prediction_edges = prediction - binary_dilation(prediction)
                ground_truth = mask - binary_dilation(mask)
                image[0, 0, ground_truth.bool()] = 1
                image[0, 1, prediction_edges.bool()] = 1
        
                axs[i//width][i%width].imshow(image[0].to('cpu').permute(1, 2, 0))
                if n_examples == i + 1:
                    break
                i += 1
        plt.show()

    def test_evaluation(self, test_dataset, test_loader, loss_fn):
        with torch.no_grad():
            running_IoU = 0
            running_dice = 0
            running_loss = 0

            for i, data in enumerate(test_loader):
                img, mask = data
                img, mask = img.to(self.device), mask.to(self.device)
                predictions = self.model(img)
                predictions = predictions.squeeze(1)
                running_dice += self.dice_pytorch(predictions, mask).sum().item()
                running_IoU += self.iou_pytorch(predictions, mask).sum().item()
                loss = loss_fn(predictions, mask)
                running_loss += loss.item() * img.size(0)

            test_loss = running_loss / len(test_dataset)
            test_dice = running_dice / len(test_dataset)
            test_IoU = running_IoU / len(test_dataset)

            print(f'Test results: Loss: {test_loss:.4f} | Mean IoU: {test_IoU:.4f} | Dice coefficient: {test_dice:.4f}')

    def create_model(self):
        model = smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation='sigmoid',
    )
        model.to(self.device)
        return model
    
    def run(self):
    
        # Fill missing values in the DataFrame with the most frequent value
        self.df = pd.DataFrame(self.imputer.fit_transform(self.df), columns=self.df.columns)
        
        # Create a DataFrame with patient ID, image filename, and mask filename
        filenames_df = pd.DataFrame((self.get_file_row(filename) for filename in FILE_PATH), 
                                columns=['Patient', 'image_filename', 'mask_filename'])

        self.df = pd.merge(self.df, filenames_df, on="Patient")

        # Split the DataFrame into training and temporary test sets
        train_df, temp_test_df = train_test_split(self.df, test_size=0.3)

        # Further split the temporary test set into the final test and validation sets
        test_df, valid_df = train_test_split(temp_test_df, test_size=0.5)

        # Define data augmentation transformations
        transform = A.Compose([
        A.ChannelDropout(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
        ])

        # Create training, validation, and test datasets
        train_dataset = MriDataset(train_df, transform)
        valid_dataset = MriDataset(valid_df)
        test_dataset = MriDataset(test_df)

        # Create data loaders for training, validation, and test datasets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)

        self.show_sample(train_dataset)
        loss_fn = self.BCE_dice
        optimizer = Adam(self.model.parameters(), lr=0.001)
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=2, factor=0.2)
        history = self.training_loop( train_loader, valid_loader, optimizer, loss_fn, lr_scheduler)
        self.show_evaluation(history)
        self.test_evaluation(  test_dataset, test_loader, loss_fn)
        self.show_testing( test_loader)


if __name__ == "__main__":
    FILES_DIR = 'data/lgg-mri-segmentation/kaggle_3m/'
    FILE_PATH = glob(os.path.join(FILES_DIR, '*/*[0-9].tif'))
    CSV_PATH = 'data/lgg-mri-segmentation/kaggle_3m/data.csv'

    EPOCHS = 1

    df = pd.read_csv(CSV_PATH)
    
    app = Main(df, EPOCHS)
    app.run()
