import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. 데이터 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


# 2. 기본 모델 정의 및 학습
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 입력 채널: 1, 출력 채널: 32
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = BasicCNN()
criterion = nn.CrossEntropyLoss()
# 기존의 SGD 옵티마이저 사용
optimizer = optim.SGD(model.parameters(), lr=0.001)


# 모델 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 역전파 및 옵티마이저 스텝
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')


# 모델 평가 함수
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# 기본 모델 학습 및 평가
train_model(model, train_loader, criterion, optimizer)
accuracy = evaluate(model, test_loader)
print(f'\nTest Accuracy of Basic Model: {accuracy:.2f}%')


# 3. 성능 향상 전략 적용
improved_model = BasicCNN()
# 새로운 Optimizer를 사용하여 성능 향상을 노림
# 수업시간에 배웠던 optimizer를 사용하면서 시도해보시길 바랍니다.
# 참고: https://pytorch.org/docs/stable/optim.html
optimizer = optim.Adam(improved_model.parameters(), lr = 0.001)

# 개선된 모델 학습 및 평가
train_model(improved_model, train_loader, criterion, optimizer)
accuracy = evaluate(improved_model, test_loader)
print(f'\nTest Accuracy of Improved Model: {accuracy:.2f}%')
