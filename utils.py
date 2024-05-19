import cv2
import torch
import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from scipy.ndimage import binary_dilation

class MRIDataset(Dataset):
    def __init__(self, df, transform=None, mean=0.5, std=0.25):
        super(MRIDataset, self).__init__()
        self.df = df
        self.transform = transform
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx, raw=False):
        row = self.df.iloc[idx]
        img = cv2.imread(row['image_filename'], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(row['mask_filename'], cv2.IMREAD_GRAYSCALE)
        if raw:
            return img, mask
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        
        img = TF.to_tensor(img)
        # Convert mask to tensor and normalize
        mask = TF.to_tensor(mask) // 255
        return img, mask
    
class EarlyStopping():
    """
    Stops training when loss stops decreasing in a PyTorch module.
    """
    def __init__(self, patience:int = 6, min_delta: float = 0, weights_path: str = 'weights.pt'):
        """
        :param patience: number of epochs of non-decreasing loss before stopping
        :param min_delta: minimum difference between best and new loss that is considered
            an improvement
        :param weights_path: Path to the file that should store the model's weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.weights_path = weights_path

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.weights_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def load_weights(self, model: torch.nn.Module):
        """
        Loads weights of the best model.
        :param model: model to which the weights should be loaded
        """
        return model.load_state_dict(torch.load(self.weights_path))

class Handler():
    def __init__(self) -> None:
        pass

    def get_file_row(self, path):
        """Produces ID of a patient, image and mask filenames from a particular path"""
        filename = os.path.basename(path)
        segments = filename.split('_')
        
        # Patient ID in the csv file consists of the first 3 filename segments
        patient_id = '_'.join(segments[:3])
        
        path_no_ext, ext = os.path.splitext(path)
        mask_filename = os.path.join(path_no_ext + '_mask' + ext)
        
        return [patient_id, path, mask_filename]
    
    def iou_pytorch(self,predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
        """Calculates Intersection over Union for a tensor of predictions"""
        predictions = torch.where(predictions > 0.5, 1, 0)
        labels = labels.byte()
        
        intersection = (predictions & labels).float().sum((1, 2))
        union = (predictions | labels).float().sum((1, 2))
        
        iou = (intersection + e) / (union + e)
        return iou

    def dice_pytorch(self, predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
        """Calculates Dice coefficient for a tensor of predictions"""
        predictions = torch.where(predictions > 0.5, 1, 0)
        labels = labels.byte()

        intersection = (predictions & labels).float().sum((1, 2))
        sum_predictions = predictions.float().sum((1, 2))
        sum_labels = labels.float().sum((1, 2))

        return ((2 * intersection) + e) / (sum_predictions + sum_labels + e)


    def BCE_dice(self,output, target, alpha=0.01):
        target = target.squeeze(1)
        bce = torch.nn.functional.binary_cross_entropy(output, target)
        soft_dice = 1 - self.dice_pytorch(output, target).mean()
        return bce + alpha * soft_dice

    def train_model(self, device, epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler):
        history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
        early_stopping = EarlyStopping(patience=7)
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            running_loss = 0
            model.train()
            for i, data in enumerate(tqdm(train_loader)):
                img, mask = data
                img, mask = img.to(device), mask.to(device)
                predictions = model(img)
                predictions = predictions.squeeze(1)
                loss = loss_fn(predictions, mask)
                running_loss += loss.item() * img.size(0)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            model.eval()
            with torch.no_grad():
                running_IoU = 0
                running_dice = 0
                running_valid_loss = 0
                for i, data in enumerate(valid_loader):
                    img, mask = data
                    img, mask = img.to(device), mask.to(device)
                    predictions = model(img)
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
            print(f'Epoch: {epoch}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss} | Validation Mean IoU: {val_IoU} '
            f'| Validation Dice coefficient: {val_dice}')
            
            lr_scheduler.step(val_loss)
            if early_stopping(val_loss, model):
                early_stopping.load_weights(model)
                break
        model.eval()
        return history
    
    def test(self, device, test_dataset, test_loader, model, loss_fn):
        with torch.no_grad():
            running_IoU = 0
            running_dice = 0
            running_loss = 0
            for i, data in enumerate(test_loader):
                img, mask = data
                img, mask = img.to(device), mask.to(device)
                predictions = model(img)
                predictions = predictions.squeeze(1)
                running_dice += self.dice_pytorch(predictions, mask).sum().item()
                running_IoU += self.iou_pytorch(predictions, mask).sum().item()
                loss = loss_fn(predictions, mask)
                running_loss += loss.item() * img.size(0)
            loss = running_loss / len(test_dataset)
            dice = running_dice / len(test_dataset)
            IoU = running_IoU / len(test_dataset)
    
        print(f'Tests: loss: {loss} | Mean IoU: {IoU} | Dice coefficient: {dice}') 
   
class Visualization():
    def __init__(self) -> None:
        pass

    def visualize_samples(self, train_dataset, n_examples=4):

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
    
    def visualize_training(self, history):
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
    
    def visualize_test(self, device, test_loader, model):
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
                image = image.to(device)
                prediction = model(image).to('cpu')[0][0]
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