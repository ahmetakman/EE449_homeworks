import torchinfo

import torch

# example mlp classifier
class mlp_1(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(mlp_1, self).__init__()
        self.input_size = input_size
        self.FC = torch.nn.Linear(input_size, hidden_size)
        self.prediction_layer = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = x.view(-1, self.input_size)
        hidden = self.FC(x)
        relu = self.relu(hidden)
        output = self.prediction_layer(relu)
        return output

# initialize your model
model_mlp_1 = mlp_1(784,32,10)


torchinfo.summary(model_mlp_1, input_size=(96, 784))

# Total params: 25,450
# Trainable params: 25,450





# example mlp classifier
class mlp_2(torch.nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super(mlp_2, self).__init__()
        self.input_size = input_size
        self.FC1 = torch.nn.Linear(input_size, hidden_size_1)
        self.FC2 = torch.nn.Linear(hidden_size_1, hidden_size_2, bias=False)
        self.prediction_layer = torch.nn.Linear(hidden_size_2, num_classes)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = x.view(-1, self.input_size)
        hidden1 = self.FC1(x)
        relu = self.relu(hidden1)
        hidden2 = self.FC2(relu)
        output = self.prediction_layer(hidden2)
        return output

# initialize your model
model_mlp_2 = mlp_2(784 ,32 ,64 ,10)

torchinfo.summary(model_mlp_2, input_size=(96, 784))


# Total params: 27,818
# Trainable params: 27,818


# example cnn_3 classifier
class cnn_3(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(cnn_3, self).__init__()
        self.input_size = input_size
        self.Conv1 = torch.nn.Conv2d(1 ,16 ,3, stride=1, padding=1)
        self.MaxPool = torch.nn.MaxPool2d(2, stride=2)
        self.Conv2 = torch.nn.Conv2d(16, 8, 5,  stride=1, padding=1)
        self.Conv3 = torch.nn.Conv2d(8, 16, 7, stride=1, padding=1)
        self.prediction_layer = torch.nn.Linear(1296, num_classes)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        hidden1 = self.Conv1(x)
        relu1 = self.relu(hidden1)
        hidden2 = self.Conv2(relu1)
        relu2 = self.relu(hidden2)
        pool = self.MaxPool(relu2)
        hidden3 = self.Conv3(pool)
        flattened = hidden3.view(96, -1)
        output = self.prediction_layer(flattened)
        return output

# initialize your model
model_cnn_3 = cnn_3(784 ,10)

torchinfo.summary(model_cnn_3, input_size=(96, 784))

# Total params: 22,626
# Trainable params: 22,626


# example cnn_4 classifier
class cnn_4(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(cnn_4, self).__init__()
        self.input_size = input_size
        self.Conv1 = torch.nn.Conv2d(1 ,16 ,3, stride=1, padding=1)
        self.Conv2 = torch.nn.Conv2d(16, 8, 3,  stride=1, padding=1)
        self.Conv3 = torch.nn.Conv2d(8, 16, 5, stride=1, padding=1)
        self.MaxPool = torch.nn.MaxPool2d(2, stride=2)
        self.Conv4 = torch.nn.Conv2d(16, 16, 5,  stride=1, padding=1)
        self.prediction_layer = torch.nn.Linear(400, num_classes)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        hidden1 = self.Conv1(x)
        relu1 = self.relu(hidden1)
        hidden2 = self.Conv2(relu1)
        relu2 = self.relu(hidden2)
        hidden3 = self.Conv3(relu2)
        relu3 = self.relu(hidden3)
        pool1 = self.MaxPool(relu3)
        hidden4 = self.Conv4(pool1)
        pool2 = self.MaxPool(hidden4)
        flattened = pool2.view(96, -1)
        output = self.prediction_layer(flattened)
        return output

# initialize your model
model_cnn_4 = cnn_4(784 ,10)

torchinfo.summary(model_cnn_4, input_size=(96, 784))

# Total params: 14,962
# Trainable params: 14,962

# example cnn_5 classifier
class cnn_5(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(cnn_5, self).__init__()
        self.input_size = input_size
        self.Conv1 = torch.nn.Conv2d(1 ,8 ,3, stride=1, padding=1)
        self.Conv2 = torch.nn.Conv2d(8, 16, 3,  stride=1, padding=1)
        self.Conv3 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.Conv4 = torch.nn.Conv2d(8, 16, 3,  stride=1, padding=1)
        self.MaxPool = torch.nn.MaxPool2d(2, stride=2)
        self.Conv5 = torch.nn.Conv2d(16, 16, 3,  stride=1, padding=1)
        self.Conv6 = torch.nn.Conv2d(16, 8, 3,  stride=1, padding=1)
        self.prediction_layer = torch.nn.Linear(392, num_classes)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        hidden1 = self.Conv1(x)
        relu1 = self.relu(hidden1)
        hidden2 = self.Conv2(relu1)
        relu2 = self.relu(hidden2)
        hidden3 = self.Conv3(relu2)
        relu3 = self.relu(hidden3)
        hidden4 = self.Conv4(relu3)
        relu4 = self.relu(hidden4)
        pool1 = self.MaxPool(relu4)
        hidden5 = self.Conv5(pool1)
        relu5 = self.relu(hidden5)
        hidden6 = self.Conv6(relu5)
        relu6 = self.relu(hidden6)
        pool2 = self.MaxPool(relu6)
        flattened = pool2.view(96, -1)
        output = self.prediction_layer(flattened)
        return output

# initialize your model
model_cnn_5 = cnn_5(784 ,10)


torchinfo.summary(model_cnn_5, input_size=(96, 784))
# Total params: 10,986
# Trainable params: 10,986