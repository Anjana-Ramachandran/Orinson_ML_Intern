import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

class ObjectDetectionModel(nn.Module):
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        self.backbone = torchvision.models.detection.yolov3(pretrained=True)

    def forward(self, x):
        return self.backbone(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = torchvision.datasets.CocoDetection(root='path/to/train', transform=transform)
val_dataset = torchvision.datasets.CocoDetection(root='path/to/val', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ObjectDetectionModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs[0]['boxes'], targets[0]['boxes'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

    model.eval()
    with torch.no_grad():
        total_correct = 0
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            total_correct += sum([1 for output, target in zip(outputs[0]['boxes'].tolist(), targets[0]['boxes'].tolist()) if output == target])
        accuracy = total_correct / len(val_loader.dataset)
        print(f'Epoch {epoch+1}, Val Accuracy: {accuracy:.4f}')

torch.save(model.state_dict(), 'object_detection_model.pth')
