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
from torch.nn.functional import cosine_similarity

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SiameseDataset(Dataset):
    def __init__(self, image_pairs, images_folder, transform=None):
        self.image_pairs = image_pairs
        self.transform = transform
        self.images_folder = images_folder

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        task_id, label = self.image_pairs[index]

        image1_path = os.path.join(self.images_folder, f"{task_id}_1.jpg")
        image2_path = os.path.join(self.images_folder, f"{task_id}_2.jpg")


        image1 = self.load_image(image1_path)
        image2 = self.load_image(image2_path)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label

    def load_image(self, image_path):
        placeholder_image_path = 'E:\\images_rooms_val\\330201107#330201147_1.jpg'

        try:
            image = Image.open(image_path)
            image.load()
            return image
        except OSError as e:
            print(f"Помилка при завантаженні зображення {image_path}: {e}. Використання замінного зображення.")
            placeholder_image = Image.open(placeholder_image_path)
            placeholder_image.load()
            return placeholder_image

class SiameseDataLoader:
    def __init__(self, train_json_path, test_json_path, images_folder_train, images_folder_test, batch_size=8, transform=None):
        self.train_json_path = train_json_path
        self.test_json_path = test_json_path
        self.images_folder_train = images_folder_train
        self.images_folder_test = images_folder_test
        self.batch_size = batch_size
        self.transform = transform

    def create_datasets(self):
        train_data, train_task_ids = self.load_data_from_json(self.train_json_path)
        test_data, test_task_ids = self.load_data_from_json(self.test_json_path)

        image_pairs_train = [(entry['taskId'], int(entry["answers"][0]["answer"][0]["id"])) for entry in train_data]
        #image_pairs_train = [(entry['taskId'], int(entry["answers"][0]["answer"][0]["id"])) for entry in
                             #train_data[:320]]
        image_pairs_test = [(entry['taskId'], 0) for entry in test_data]

        train_dataset = SiameseDataset(image_pairs_train, self.images_folder_train, transform=self.transform)
        test_dataset = SiameseDataset(image_pairs_test, self.images_folder_test, transform=self.transform)

        return train_dataset, test_dataset, train_task_ids, test_task_ids

    def create_data_loaders(self):
        train_dataset, test_dataset, _, test_task_ids = self.create_datasets()

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, test_task_ids

    def load_data_from_json(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)["data"]["results"]
        task_ids = [item["taskId"] for item in data]
        return data, task_ids

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        #self.backbone = models.vgg16(pretrained=True)
        #self.embedding_size = self.backbone.classifier[-1].in_features
        #self.backbone.classifier = nn.Identity()
        self.backbone = models.resnet50(pretrained=True)
        self.embedding_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

    def forward(self, x1, x2):
        embedding1 = self.backbone(x1)
        embedding2 = self.backbone(x2)
        return embedding1, embedding2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, target):
        euclidean_distance = nn.functional.pairwise_distance(embedding1, embedding2)
        loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                       target * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class SiameseNetworkTrainer:
    def __init__(self, network, train_loader, test_loader, margin=1.0, lr=0.0001, threshold=70):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = network.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = ContrastiveLoss(margin).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.threshold = threshold / 100

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for batch_idx, (images1, images2, labels) in enumerate(self.train_loader):
                images1, images2, labels = images1.to(self.device), images2.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                embeddings1, embeddings2 = self.model(images1, images2)
                loss = self.criterion(embeddings1, embeddings2, labels.float())
                loss.backward()
                self.optimizer.step()

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")

    def test(self):
        true_positives = []
        false_positives = []
        false_negatives = []
        result = []

        with torch.no_grad():
            for image1, image2, label in self.test_loader:
                image1, image2, label = image1.to(self.device), image2.to(self.device), label.to(self.device)

                output1, output2 = self.model(image1, image2)
                euclidean_distance = nn.functional.pairwise_distance(output1, output2)
                predictions = (euclidean_distance < self.threshold).float()

                labels_np = label.cpu().numpy()
                predictions_np = predictions.cpu().numpy()

                for pred, lbl in zip(predictions_np, labels_np):
                    result.append(int(pred))
                    if pred == 1 and lbl == 1:
                        true_positives.append(pred)
                    elif pred == 1 and lbl == 0:
                        false_positives.append(pred)
                    elif pred == 0 and lbl == 1:
                        false_negatives.append(pred)


        return true_positives, false_positives, false_negatives, result

    def get_csv(self, task_ids, result):
        with open('output_val2.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
            csvwriter.writerow(['taskId', 'answer'])
            for res, task_id in zip(result, task_ids):
                csvwriter.writerow([task_id, res])


def main():
    network = SiameseNetwork()

    transform = transforms.Compose([
        Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        Resize((224, 224)),
        ToTensor()
    ])

    siamese_data_loader = SiameseDataLoader(
        train_json_path='train_task1.json',
        test_json_path='val_task1.json',
        images_folder_train='E:\\images_rooms_train',
        images_folder_test='E:\\images_rooms_val',
        batch_size=8,
        transform=transform
    )

    train_loader, test_loader, test_task_ids  = siamese_data_loader.create_data_loaders()

    trainer = SiameseNetworkTrainer(network=network, train_loader=train_loader, test_loader=test_loader)

    trainer.train(num_epochs=2)

    true_positives, false_positives, false_negatives, result = trainer.test()

    trainer.get_csv(test_task_ids, result)

    #precision = len(true_positives) / (len(true_positives) + len(false_positives))
    #recall = len(true_positives) / (len(true_positives) + len(false_negatives))
    #f1_score = 2 * precision * recall / (precision + recall)

    #print(f"Precision: {precision}")
    #print(f"Recall: {recall}")
    #print(f"F1 Score: {f1_score}")


if __name__ == '__main__':
    main()

