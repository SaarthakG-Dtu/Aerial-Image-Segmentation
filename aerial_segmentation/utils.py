import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("outputs", "matplotlib"))

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms as T
import torch.nn.functional as F

from os.path import isfile, join
from os import listdir

from .model_factory import N_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, true):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        true_one_hot = F.one_hot(true, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_one_hot, dims)
        cardinality = torch.sum(probs + true_one_hot, dims)
        
        dice_score = (2. * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - torch.mean(dice_score)

class DiceCELoss(nn.Module):
    """
    Combined Dice and Cross Entropy Loss
    """
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(DiceCELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss

def get_image_id_df(root_img_path):
    """
    Generate a DataFrame containing the image IDs.

    Args:
        root_img_path (str): Path to the directory containing the images.

    Returns:
        pd.DataFrame: A DataFrame containing the image IDs.

    """
    name = []
    filenames = [f for f in listdir(root_img_path) if isfile(join(root_img_path, f))]
    for filename in filenames:
        name.append(filename.split(".")[0])
    return pd.DataFrame({"id": name}, index=np.arange(0, len(name)))



def pixel_accuracy(predicted_image, mask):
    """
    Calculate the pixel accuracy between the predicted image and the ground truth mask.

    Args:
        predicted_image (torch.Tensor): Predicted image tensor of shape (N, C, H, W).
        mask (torch.Tensor): Ground truth mask tensor of shape (N, H, W).

    Returns:
        float: Pixel accuracy between the predicted image and the ground truth mask.

    """
    with torch.no_grad():
        # Convert predicted_image to class predictions
        predicted_image = torch.argmax(F.softmax(predicted_image, dim=1), dim=1)

        # Compare predicted_image with mask to get pixel-wise correctness
        correct = torch.eq(predicted_image, mask).int()

        # Calculate pixel accuracy
        accuracy = float(correct.sum()) / float(correct.numel())

    return accuracy


def mean_iou(predicted_label, label, eps=1e-10, num_classes=N_CLASSES):
    """
    Calculate the mean Intersection over Union (IoU) between the predicted labels and the ground truth labels.

    Args:
        predicted_label (torch.Tensor): Predicted label tensor of shape (N, C, H, W).
        label (torch.Tensor): Ground truth label tensor of shape (N, H, W).
        eps (float, optional): Epsilon value for numerical stability.
        num_classes (int, optional): Number of classes.

    Returns:
        float: Mean IoU value.

    """
    with torch.no_grad():
        # Convert predicted_label to class predictions
        predicted_label = F.softmax(predicted_label, dim=1)
        predicted_label = torch.argmax(predicted_label, dim=1)

        # Reshape predicted_label and label for easier computation
        predicted_label = predicted_label.contiguous().view(-1)
        label = label.contiguous().view(-1)

        iou_single_class = []
        for class_number in range(0, num_classes):
            true_predicted_class = predicted_label == class_number
            true_label = label == class_number

            if true_label.long().sum().item() == 0:
                iou_single_class.append(np.nan)
            else:
                # Calculate intersection and union
                intersection = (
                    torch.logical_and(true_predicted_class, true_label)
                    .sum()
                    .float()
                    .item()
                )
                union = (
                    torch.logical_or(true_predicted_class, true_label)
                    .sum()
                    .float()
                    .item()
                )

                # Calculate IoU for the current class
                iou = (intersection + eps) / (union + eps)
                iou_single_class.append(iou)

        # Calculate mean IoU across all classes
        return np.nanmean(iou_single_class)


#### Some Plotting Function ####

def plot_loss_vs_epoch(history, save_path=None):
    """
    Plot the training and validation loss versus epochs.

    Args:
        history (dict): Dictionary containing the training history with keys 'val_loss' and 'train_loss'.

    """
    plt.plot(history["val_loss"], label="val_loss", marker="o")
    plt.plot(history["train_loss"], label="Train loss", marker="o")
    plt.title("Loss per epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_iou_score_vs_epoch(history, save_path=None):
    """
    Plot the training and validation mean IoU scores versus epochs.

    Args:
        history (dict): Dictionary containing the training history with keys 'train_miou' and 'val_miou'.

    """
    plt.plot(history["train_miou"], label="Train mIoU", marker="*")
    plt.plot(history["val_miou"], label="Val mIoU", marker="*")
    plt.title("mIoU Score per Epoch ")
    plt.ylabel("mean IoU")
    plt.xlabel("epoch")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_accuracy_vs_epoch(history, save_path=None):
    """
    Plot the training and validation accuracy versus epochs.

    Args:
        history (dict): Dictionary containing the training history with keys 'train_acc' and 'val_acc'.

    """
    plt.plot(history["train_acc"], label="Train Accuracy", marker="*")
    plt.plot(history["val_acc"], label="Val Accuracy", marker="*")
    plt.title("Accuracy vs Epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def predict_image_mask_miou(
    model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    """
    Predict the mask for an input image using a trained model and calculate the mean IoU score.

    Args:
        model (torch.nn.Module): Trained model.
        image (PIL.Image.Image): Input image.
        mask (torch.Tensor): Ground truth mask.
        mean (list, optional): Mean values for image normalization.
        std (list, optional): Standard deviation values for image normalization.

    Returns:
        torch.Tensor: Predicted mask.
        float: Mean IoU score.

    """
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        predicted_image = model(image)
        mean_iou_score = mean_iou(predicted_image, mask)
        masked = torch.argmax(predicted_image, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, mean_iou_score


def predict_iamge_mask_pixel_accuracy(
    model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    """
    Predict the mask for an input image using a trained model and calculate the pixel accuracy.

    Args:
        model (torch.nn.Module): Trained model.
        image (PIL.Image.Image): Input image.
        mask (torch.Tensor): Ground truth mask.
        mean (list, optional): Mean values for image normalization.
        std (list, optional): Standard deviation values for image normalization.

    Returns:
        torch.Tensor: Predicted mask.
        float: Pixel accuracy.

    """
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        predicted_image = model(image)
        acc = pixel_accuracy(predicted_image, mask)
        masked = torch.argmax(predicted_image, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc


def miou_score_from_trained_model(model, test_set):
    """
    Calculate the mean IoU scores for a trained model on a test dataset.

    Args:
        model (torch.nn.Module): Trained model.
        test_set (torch.utils.data.Dataset): Test dataset.

    Returns:
        list: List of mean IoU scores for each sample in the test dataset.

    """
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou


def pixel_accuracy_from_trained_model(model, test_set):
    """
    Calculate the pixel accuracy for a trained model on a test dataset.

    Args:
        model (torch.nn.Module): Trained model.
        test_set (torch.utils.data.Dataset): Test dataset.

    Returns:
        list: List of pixel accuracy values for each sample in the test dataset.

    """
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_iamge_mask_pixel_accuracy(model, img, mask)
        accuracy.append(acc)
    return accuracy
