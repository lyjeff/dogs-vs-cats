import torch
import torch.nn as nn
from torchvision.models import vgg19, resnet50, densenet161
# from torchsummary import summary

def VGG19():
    model = vgg19(pretrained=True)

    # 把參數凍結
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.classifier[6] = nn.Linear(4096, 2)

    return model


def VGG19_2():
    model = vgg19(pretrained=True)

    # 把參數凍結
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[3] = nn.Linear(4096, 4096)
    model.classifier[6] = nn.Linear(4096, 2)

    return model


def ResNet():
    model = resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # 修改全連線層的輸出
    model.fc = nn.Linear(2048, 2)

    return model


def Densenet():
    model = densenet161()

    for param in model.parameters():
        param.requires_grad = False

    # 修改全連線層的輸出
    model.classifier = nn.Linear(2208, 2)

    return model


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        ################################ you need to modify the cnn model here ################################

        # after convolutoin, the feature map size = ((origin + padding*2 - kernel_size) / stride) + 1
        # input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)   # ((224+2*1-3)/1)+1=224  # output_shape=(64,224,224)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # output_shape=(64,112,112) # (224)/2

        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)   # ((224+2*1-3)/1)+1=224  # output_shape=(128,224,224)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)    # output_shape=(128,56,56)

        self.cnn3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)   # output_shape=(256,112,112)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)  # output_shape=(256,28,28)

        self.cnn4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)   # output_shape=(512,28,28)
        self.relu4 = nn.ReLU()
        self.cnn5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)   # output_shape=(512,28,28)
        self.relu5 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)  # output_shape=(512,14,14)

        self.cnn6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)   # output_shape=(512,14,14)
        self.relu6 = nn.ReLU()
        self.cnn7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)   # output_shape=(512,14,14)
        self.relu7 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)    # output_shape=(512,7,7)

        self.fc1 = nn.Linear(512*7*7, 1024)
        self.relu8 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu9 = nn.ReLU()
        self.fc3 = nn.Linear(512, 2)
        # =================================================================================================== #

    def forward(self, x):

        ################################ you need to modify the cnn model here ################################
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)

        out = self.cnn4(out)
        out = self.relu4(out)
        out = self.cnn5(out)
        out = self.relu5(out)
        out = self.maxpool4(out)

        out = self.cnn6(out)
        out = self.relu6(out)
        out = self.cnn7(out)
        out = self.relu7(out)
        out = self.maxpool5(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu8(out)
        out = self.fc2(out)
        out = self.relu9(out)
        out = self.fc3(out)
        # =================================================================================================== #

        return out


def model_builder(model_name):

    # load model
    if model_name == "VGG19":
        model = VGG19()
    elif model_name == "VGG19_2":
        model = VGG19_2()
    elif model_name == "MyCNN":
        model = MyCNN()
    elif model_name == "ResNet":
        model = ResNet()
    elif model_name == "Densenet":
        model = Densenet()
    else:
        model = VGG19()

    return model


if __name__ == '__main__':
    # model = Densenet()
    # summary(model, input=(3,224,224), device="cpu")
    # print(model)
    pass