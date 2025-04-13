import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import os
import argparse

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data', help='Dataset directory')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--model_name', type=str, default='vgg16', choices=['vgg16', 'resnet50'])
args = parser.parse_args()

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
def load_datasets(data_dir):
    train_dataset = ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    val_dataset = ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=test_transform
    )
    test_dataset = ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=test_transform
    )
    return train_dataset, val_dataset, test_dataset

# 处理类别不平衡
def get_class_weights(dataset):
    class_counts = torch.bincount(torch.tensor(dataset.targets))
    class_weights = 1. / class_counts.float()
    return class_weights

# 构建模型
def build_model(model_name, num_classes, pretrained=True):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    best_acc = 0.0
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # 验证
        val_epoch_loss, val_epoch_acc = evaluate(model, val_loader, criterion)
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        # 保存最佳模型
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}\n')
    
    # 绘制训练曲线
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.close()

# 评估函数
def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    
    # 分类报告和混淆矩阵
    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0,1], test_loader.dataset.classes)
    plt.yticks([0,1], test_loader.dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    # 加载数据
    train_dataset, val_dataset, test_dataset = load_datasets(args.data_dir)
    
    # 处理类别不平衡
    class_weights = get_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 初始化模型
    model = build_model(args.model_name, num_classes=len(train_dataset.classes))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 训练模型
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs)
    
    # 测试最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')