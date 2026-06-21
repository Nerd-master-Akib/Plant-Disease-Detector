from pyclbr import Class

import torch 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader, random_split 


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root = "data/train/plantvillage dataset/color",
    transform = transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset)) 

from torch.utils.data import random_split
train_size = int(0.7*len(dataset))
val_size = int(0.15*len(dataset))
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(
    dataset, [train_size, val_size, test_size]
)

print("train:", len(train_data))
print("validation:", len(val_data))
print("test:", len(test_data))


from torch.utils.data import DataLoader


train_loader = DataLoader(
    train_data,
    batch_size = 16,
    shuffle = True
)


val_loader = DataLoader(
    val_data,
    batch_size = 16,
    shuffle = False
)

test_loader = DataLoader(
    test_data,
    batch_size = 16,
    shuffle = False
)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)

print(dataset.class_to_idx) 

## First CNN codes: 

import torch.nn as nn

class PlantCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 32,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                padding = 1
            ),
            
            
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128*28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 5)) 
        
    def forward(self,x):
        x = self.conv_layers(x)
        
        x = torch.flatten(x, start_dim = 1)
        
        x = self.fc_layers(x)
        
        return x
    
model = PlantCNN()
print(model)

sample_batch,_ = next(iter(train_loader))
output = model(sample_batch)
print(output.shape) 

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

## First Training Loop

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images,labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct/total
        print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {running_loss:.4f}"
        f"Accuracy: {accuracy:.2f}%"
    )
        
torch.save(model.state_dict(), "plant_cnn.pth")
print("Model saved!")     
        
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images,labels in val_loader:
        outputs = model(images)
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")

model.load_state_dict(torch.load("plant_cnn.pth"))
model.eval()

def predict_image(image,model):
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output,1)
        return predicted 