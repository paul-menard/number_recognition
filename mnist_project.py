import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

data_train = datasets.MNIST(root = "./data", train=True, download=True, transform=transform)
data_test = datasets.MNIST(root="./data", train = False, download=True, transform=transform)

train_loader = DataLoader(dataset=data_train, batch_size=64, shuffle = True)
test_loader = DataLoader(dataset=data_test, batch_size=64, shuffle=False)

for images, labels in train_loader:
    print(images.shape)
    break 

class ChiffreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Flatten()
        self.layer1 = nn.Linear(in_features=784, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=10)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.layer0(x)
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
model = ChiffreNet()
optimizer = optim.Adam(params=model.parameters(), lr = 0.001)
loss = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()

    mean_loss = 0
    for batch in train_loader:
        images, labels = batch
        labels_hat = model.forward(images).squeeze()
        lost = loss(labels_hat, labels)

        optimizer.zero_grad()
        lost.backward()

        optimizer.step()

        mean_loss += lost
    print(mean_loss/(5*64))


# --- ÉTAPE 5 : ÉVALUATION ---
print("\n📝 Début du Test...")
model.eval() # Important : On gèle le modèle (pas d'entraînement)

correct_predictions = 0
total_predictions = 0

with torch.no_grad(): # On désactive le calcul des gradients (économie de RAM)
    for images, labels in test_loader:
        # 1. Forward
        outputs = model(images)
        
        # 2. Décision (Quel est le chiffre avec le plus haut score ?)
        # torch.max renvoie (valeur_max, index_max). On veut l'index.
        _, predicted = torch.max(outputs, 1)
        
        # 3. Comptabilité
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

accuracy = (correct_predictions / total_predictions) * 100
print(f"🏆 Précision Finale : {accuracy:.2f}%")




