import csv
import io
import json
import os

import numpy as np
import requests
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from torchvision.transforms import Lambda
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

THRESHOLD = 55
COLOR_CHANNEL_COUNT = 3
CSV_ROW_LENGTH = 22
ROOT_RANK = 0
TRAIN_DATA = "train_task1.json"
TEST_DATA = "test_task1.json"
VALIDATION_DATA = "val_task1.json"

class SiameseDataset(Dataset):
    def __init__(self, image_pairs, images_folder, transform=None):
        self.image_pairs = image_pairs
        self.transform = transform
        self.images_folder = images_folder

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        task_id, label = self.image_pairs[index]

        image1_path = os.path.join(self.images_folder, f"{task_id}_1.jpg") # Або .png, залежно від формату
        image2_path = os.path.join(self.images_folder, f"{task_id}_2.jpg")


        # Завантаження локальних зображень
        image1 = self.load_image(image1_path)
        image2 = self.load_image(image2_path)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label

    def load_image(self, image_path):
        # Стандартне зображення як заповнювач
        placeholder_image_path = 'E:\\images_rooms_val\\330201107#330201147_1.jpg'

        try:
            image = Image.open(image_path)
            image.load()  # Спробуйте завантажити зображення
            return image
        except OSError as e:
            print(f"Помилка при завантаженні зображення {image_path}: {e}. Використання замінного зображення.")
            # Використовуємо замінне зображення
            placeholder_image = Image.open(placeholder_image_path)
            placeholder_image.load()
            return placeholder_image

# Load test data from test_task1.json
test_data_path = "val_task1.json"
with open(test_data_path, "r") as file:
    test_data = json.load(file)["data"]["results"]
    task_ids = [item["taskId"] for item in test_data]

train_data_path = "train_task1.json"
with open(train_data_path, "r") as file:
    train_data = json.load(file)["data"]["results"]

# Create list of image pairs and labels
image_pairs = []
y_true = []  # List to store true class labels
for entry in test_data:
    task_id = entry['taskId']
    label = 0
    image_pairs.append((task_id, label))
    y_true.append(label)  # Append true class label

# Create list of image pairs and labels
image_pairs_train = []
for entry in train_data:
    task_id = entry['taskId']
    label = int(entry["answers"][0]["answer"][0]["id"])
    image_pairs_train.append((task_id, label))


# Define transformations
#transform = Resize((224, 224))  # Resize images to fit into ResNet50 input size
#transform = transforms.Compose([transform, ToTensor()])  # Convert PIL images to tensors

# Визначення трансформацій
transform = transforms.Compose([
    Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    Resize((224, 224)),  # Змінити розмір зображень до вхідного розміру ResNet50
    ToTensor()  # Конвертувати зображення PIL в тензори
])

# Create dataset and DataLoader
batch_size = 8
path_folder_test = 'E:\\images_rooms_val'
test_dataset = SiameseDataset(image_pairs, path_folder_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

path_folder_train = 'E:\\images_rooms_train'
train_dataset = SiameseDataset(image_pairs_train, path_folder_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define Siamese network architecture
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.embedding_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

    def forward(self, x1, x2):
        embedding1 = self.backbone(x1)
        embedding2 = self.backbone(x2)
        return embedding1, embedding2

# Define contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, target):
        euclidean_distance = nn.functional.pairwise_distance(embedding1, embedding2)
        loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                       target * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# Instantiate Siamese network, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for batch_idx, (images1, images2, labels) in enumerate(train_loader):
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings1, embeddings2 = model(images1, images2)
        loss = criterion(embeddings1, embeddings2, labels.float())
        loss.backward()
        optimizer.step()

        # Print training progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

y_pred = []
# Initialize lists to store true positives, false positives, and false negatives
true_positives = []
false_positives = []
false_negatives = []
result = []


# Iterate over test data
# Iterate over test data

for i, (image1, image2, label) in enumerate(test_loader):
    # Move tensors to the correct device
    image1 = image1.to(device)
    image2 = image2.to(device)
    #label = label.to(device)

    # Forward pass
    output1, output2 = model(image1, image2)

    # Compute Euclidean distance
    euclidean_distance = nn.functional.pairwise_distance(output1, output2)

    # Compute predictions: if distance < threshold, images are similar (label 1), else they are dissimilar (label 0)
    predictions = (euclidean_distance < THRESHOLD / 100).float()

    # Append the predictions to y_pred
    y_pred.extend(predictions.cpu().numpy())  # Extend the list with predictions

    # Convert labels and predictions to numpy arrays
    #labels_np = label.cpu().numpy()
    predictions_np = predictions.cpu().numpy()

    # Calculate true positives, false positives, and false negatives
    #for pred, lbl in zip(predictions_np, labels_np):
    #    result.append(int(pred))
    #   if pred == 1 and lbl == 1:
    #        true_positives.append(pred)
    #   elif pred == 1 and lbl == 0:
    #        false_positives.append(pred)
    #    elif pred == 0 and lbl == 1:
    #        false_negatives.append(pred)
    # Calculate true positives, false positives, and false negatives
    for pred in predictions_np:
        result.append(int(pred))


with open('output_val', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['taskId', 'answer'])
    for res, task_id in zip(result, task_ids):
        csvwriter.writerow([task_id, res])


# Calculate precision, recall, and F1 score
#precision = len(true_positives) / (len(true_positives) + len(false_positives))
#recall = len(true_positives) / (len(true_positives) + len(false_negatives))
#f1_score = 2 * precision * recall / (precision + recall)

#print(f"Precision: {precision}")
#print(f"Recall: {recall}")
#print(f"F1 Score: {f1_score}")


