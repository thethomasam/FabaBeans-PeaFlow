import click
import cv2
import os
import pandas as pd
import sys
from PIL import Image

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from sklearn.metrics import precision_score, recall_score, roc_curve, auc


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
# from peatrain import confusion_matrix_heatmap

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
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]  # Assuming the image path is in the first column
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx, 1])  # Assuming the label is in the second column

        if self.transform:
            image = self.transform(image)

        # Ensure the image tensor is in the expected shape (C x H x W)
        if image.dim() == 3 and image.shape[0] != 3:
            image = image.permute(1, 2, 0)  # Permute dimensions to match H x W x C

        return image, label # Transpose channels to the first dimension

# Neural Network model definition]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution2d_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.3)
        self.output = nn.Sequential(
            nn.Linear(in_features=1024, out_features=2)
            # Note that PyTorch works on unnormalised logits with
            # cross entropy loss, so we don't need a softmax layer.
        )
    
    def forward(self, x):
        x = self.convolution2d_stack(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.output(x)
        return x



class ImageDataset(Dataset):
    def __init__(self, annotations_file):
        self.image_labels = pd.read_csv(annotations_file)
    
    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        image_path = self.image_labels.iloc[idx, 0]
        image = cv2.imread(image_path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = Compose([ToTensor()])  # Normalises to 0-1
        image = transform(image)
        label = self.image_labels.iloc[idx, 1]
        return image, label

    


@click.command()
@click.argument("annotations_path")
@click.argument("out_dir_path")
@click.argument("isResnet")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose mode.")
@click.version_option("1.0.0", message="%(version)s")

def main(annotations_path: str, out_dir_path: str, verbose: bool,isResnet: bool,transformer: bool):
    annotations_path="/"+annotations_path
    train_network(annotations_path,out_dir_path,isResnet,transformer)

# def main(annotations_path: str, out_dir_path: str, verbose: bool):
#     train_network(annotations_path,out_dir_path,epochs=5,learning_rate=0.01)
def draw_diagnostic_curve(diagnostics,conf_matrix):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1) 
    plt.plot(diagnostics['epoch'], diagnostics['accuracy'], '-b', label='accuracy')
    plt.plot(diagnostics['epoch'], diagnostics['loss'], '-r', label='loss')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title('Accuracy Curve')

    # save image
    plt.savefig("accuracy_curve.png")  
    # should before show method
    

    plt.subplot(1, 2, 2) 
    # confusion_matrix_heatmap(conf_matrix, ['plant','not plant'])
 # 1 row, 2 columns, second plot
    

    plt.tight_layout()  # To prevent overlapping of labels, titles, etc.
    plt.show()

# Call the function to display the combined plots







    
    




def train_network(annotations_path,out_dir_path,epochs=5,learning_rate=0.001,isResnet=True,transformer=False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isResnet:
        print("ResNet")
        name='Resnet-trainer'
        model = models.resnet18(pretrained=True).to(device)
        for param in model.parameters():
            param.requires_grad = False
        num_classes = 2 # Replace with the number of classes in your dataset
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        loss_fn = nn.CrossEntropyLoss()
        #optimiser = torch.optim.SGD(model.fc.parameters(), lr=0.0001, momentum=0.9)
        optimiser = torch.optim.Adam(model.fc.parameters(), lr=0.001)
        images = ImageDataset(annotations_path)
        train_validation_split = [0.8, 0.3]
        batch_size=64
        images = random_split(images, lengths=train_validation_split)
        training_data, validation_data = images
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        validation_dataloader = DataLoader(validation_data, batch_size=batch_size)
    
    else:    
        print("CNN-trainer")
        name='CNN-trainer'
        model = NeuralNetwork().to(device)
        
        loss_fn = nn.CrossEntropyLoss()
        optimiser = optim.RMSprop(model.parameters(), lr=0.001)
        batch_size = 64
        train_validation_split = [0.8, 0.2]
        print(model)
        print(annotations_path)
        images = ImageDataset(annotations_path)
        images = random_split(images, lengths=train_validation_split)
        training_data, validation_data = images
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

    
    
    all_loss = pd.DataFrame({"epoch": [], "train_loss": [], "val_loss": []})
    acc_loss = pd.DataFrame({"epoch": [], "train_accuracy": [], "val_accuracy": []})


    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")

        # Train
        size = len(train_dataloader.dataset)
        model.train()
        count = 0
        correct_train = 0 
        total_train_loss = 0.0
        for batch, (X, y) in enumerate(train_dataloader):
            # optimiser.zero_grad()
           
            # outputs = model(X)
            # loss = loss_fn(outputs.logits, y)
            # loss.backward()
            # optimiser.step()
            # X=X.to(device)
            # y=y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            loss = loss.item()
            count += len(X)
            total_train_loss += loss
            correct_train += (pred.argmax(dim=1) == y).sum().item()  # Count correct predictions

        # print(f"loss: {loss:>7f}  [{count:>4d}/{size:>4d}]")

    # Calculate training accuracy
        train_accuracy = correct_train / size

    # Validate (and compute accuracy/loss estimates)
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        size = len(validation_dataloader.dataset)
        num_batches = len(validation_dataloader)
        model.eval()
        val_loss, correct = 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X, y in validation_dataloader:
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= num_batches
        val_accuracy = correct / size
            
        print(f"Validation:\nAccuracy:{val_accuracy:>7f}, val_loss:{loss:>7f}\n")
        acc_loss.loc[len(acc_loss)] = [epoch + 1, train_accuracy, val_accuracy]
        all_loss.loc[len(all_loss)]=[epoch+1,avg_train_loss,val_loss]

    # Save training accuracy/loss values
    precision = [precision_score(all_labels, np.argmax(all_preds, axis=1), average='macro')]
    recall = [recall_score(all_labels, np.argmax(all_preds, axis=1), average='macro')]

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, np.argmax(all_preds, axis=1))
    roc_auc = auc(fpr, tpr)

    metrics_df = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'ROC_AUC': roc_auc
})

    acc_loss_filename = name+"acc_loss.csv"
    all_loss_filename=name+"all_loss.csv"
    metric_df_name=name+"metrics"
    metrics_df.to_csv(metric_df_name  , index=False)
    acc_loss.to_csv(acc_loss_filename, index=False)
    all_loss.to_csv(all_loss_filename, index=False)
    print(f"Saved running train-val/loss recording to {acc_loss_filename}")
    print(f"Saved running train-val/accuracy recording to {all_loss_filename}")


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(all_loss['train_loss'], label='Train Loss')
    plt.plot(all_loss['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')

    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc_loss['train_accuracy'], label='Train Accuracy')
    plt.plot(acc_loss['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    
    plt.title('Accuracy and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # Save output model (as .pth for PyTorch models)
    if isResnet:
        model_filename = "RESNET_nnmodel.pth"
    elif transformer:
        model_filename = "transformer_nnmodel.pth"
    else:
        model_filename = "cnnmodel.pth"


    torch.save(model.state_dict(), model_filename)
    print(f"Saved neural network model to {model_filename}")
        

if __name__ == "__main__":
    main()
 