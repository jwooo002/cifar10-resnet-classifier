import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # 그래프를 그리기 위해 추가
import logging                 # 로깅을 위해 추가

# 1. 설정 및 로깅 설정
# ---------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 하이퍼파라미터
num_epochs = 20
batch_size = 128
learning_rate = 0.001

# --- 로깅 설정 시작 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt"), # 로그 파일 생성
        logging.StreamHandler()                  # 콘솔에도 출력
    ]
)
# --- 로깅 설정 끝 ---

# 2. 데이터 준비 (CIFAR-10)
# ---------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 3. 모델 정의 (ResNet-18)
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.to(device)

# 4. 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

history = {
    'train_loss': [],
    'test_loss': [],
    'test_acc': []
}


# 5. 모델 훈련
logging.info("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 에폭마다 훈련 손실 기록
    epoch_train_loss = running_loss / len(train_loader)
    history['train_loss'].append(epoch_train_loss)

    # 6. 모델 평가
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 에폭마다 테스트 손실과 정확도 기록
    epoch_test_loss = test_loss / len(test_loader)
    epoch_test_acc = 100 * correct / total
    history['test_loss'].append(epoch_test_loss)
    history['test_acc'].append(epoch_test_acc)

    # 로그 기록
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {epoch_test_acc:.2f} %')


logging.info('Finished Training')

# 7. 훈련된 모델 저장
torch.save(model.state_dict(), 'resnet18_cifar10.pth')
logging.info("Model saved to resnet18_cifar10.pth")

plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 정확도 그래프
ax1.plot(history['test_acc'], label='Test Accuracy')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Accuracy')
ax1.legend(loc='lower right')

# 손실 그래프
ax2.plot(history['train_loss'], label='Train Loss')
ax2.plot(history['test_loss'], label='Test Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Model Loss')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('training_plots.png')
logging.info("Training plots saved to training_plots.png")
