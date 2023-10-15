import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from peatrain import confusion_matrix_heatmap
from sklearn.metrics import precision_score, recall_score, roc_curve, auc

import torch
import torch.nn as nn
import torchvision.models as models

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from timm.models import deit_base_patch16_224
from transformers import AutoImageProcessor, DeiTForImageClassification
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import vision_transformer
# Neural Network model definition]


class FineTunedDeiT(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedDeiT, self).__init__()
        self.deit = deit_base_patch16_224(pretrained=True)  # Load the pre-trained DeiT model
        num_features = self.deit.head.in_features
        self.deit.head = nn.Linear(num_features, num_classes)  # Replace the classification head

    def forward(self, x):
        return self.deit(x)
# Define class labels

from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]  # Assuming the first column contains image paths
        image = Image.open(image_path)  # Open image using PIL
        label = int(self.data.iloc[idx, 1])  # Assuming the second column contains labels

        if self.transform:
            image = self.transform(image)

        return image, label

def train_network(annotations_path,out_dir_path,epochs=5,learning_rate=0.0001,isResnet=True,transformer=False):
# Load and preprocess images from the CSV
    model = FineTunedDeiT(num_classes=2)

# Define class labels
    class_labels = ['No Plant', 'Plant']

    # Load CSV with image paths
    csv_path = annotations_path
    df = pd.read_csv(csv_path)
    acc_loss = pd.DataFrame({"epoch": [], "accuracy": [], "loss": []})
    all_loss = pd.DataFrame({"epoch": [], "train_loss": [], "val_loss": []})

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    # Define a custom dataset
    images = CustomDataset(csv_file=csv_path, transform=transform)
    batch_size =32
    train_validation_split = [0.8, 0.2]
    # Split dataset into training and validation sets
    images = random_split(images, lengths=train_validation_split)
    training_data, validation_data = images

    # Create dataloaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")

        # Train
        size = len(train_dataloader.dataset)
        model.train()
        count = 0
        total_train_loss = 0.0
        correct_train = 0 
        for batch, (X, y) in enumerate(train_dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.item()
            count += len(X)
            total_train_loss += loss

            print(f"loss: {loss:>7f}  [{count:>4d}/{size:>4d}]")
            correct_train += (pred.argmax(dim=1) == y).sum().item()  # Count correct predictions


        # Validate (and compute accuracy/loss estimates)
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = correct_train / size
        
        size = len(validation_dataloader.dataset)
        num_batches = len(validation_dataloader)
        model.eval()
        val_loss, correct = 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X, y in validation_dataloader:
                pred = model(X)
                all_preds.extend(pred.cpu().numpy())  # Collect predictions
                all_labels.extend(y.cpu().numpy())
                val_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()
        val_loss /= num_batches
        val_accuracy = correct / size
            
        print(f"Validation:\nAccuracy:{train_accuracy:>7f}, loss:{loss:>7f}\n")
        acc_loss.loc[len(acc_loss)] = [epoch + 1, train_accuracy, val_accuracy]
        all_loss.loc[len(all_loss)]=[epoch+1,avg_train_loss,val_loss]
    

    # Save training accuracy/loss values
    acc_loss_filename = "acc_loss.csv"
    all_loss_filename="all_loss.csv"

    precision = [precision_score(all_labels, np.argmax(all_preds, axis=1), average='macro')]
    recall = [recall_score(all_labels, np.argmax(all_preds, axis=1), average='macro')]
    fpr, tpr, _ = roc_curve(all_labels, np.argmax(all_preds, axis=1))
    roc_auc = [auc(fpr, tpr)]
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, np.argmax(all_preds, axis=1))
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    metrics_df = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'ROC_AUC': roc_auc
})
    
    acc_loss_filename = 'transformer'+"acc_loss.csv"
    all_loss_filename='transformer'+"all_loss.csv"
    metric_df_name='transformer'+"metrics"
    metrics_df.to_csv(metric_df_name, index=False)
    acc_loss.to_csv(acc_loss_filename, index=False)
    all_loss.to_csv(all_loss_filename, index=False)
    print(f"Saved running accuracy/loss recording to {acc_loss_filename}")
    print(f"Saved running accuracy/loss recording to {all_loss_filename}")


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(all_loss['train_loss'], label='Train Loss')
    plt.plot(all_loss['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc_loss['loss'], label='Validation Loss')
    plt.plot(acc_loss['accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Accuracy and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    model_filename = "transformer_nnmodel.pth"
    torch.save(model.state_dict(), '')
    print(f"Saved neural network model to {transformer_nnmodel.pth}")
    # acc_loss.to_csv(acc_loss_filename, index=False)
    # all_loss.to_csv(all_loss_filename, index=False)
    print(f"Saved running accuracy/loss recording to {acc_loss_filename}")
    print(f"Saved running accuracy/loss recording to {all_loss_filename}")
