from collections import namedtuple
import torch
from torchvision import models as tv
from pretrainedmodels.models.inceptionv4 import InceptionV4, inceptionv4


class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple("SqueezeOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7'])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)

        return out


class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class resnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, num=50):
        super(resnet, self).__init__()
        if (num == 18):
            self.net = tv.resnet18(pretrained=pretrained)
        elif (num == 34):
            self.net = tv.resnet34(pretrained=pretrained)
        elif (num == 50):
            self.net = tv.resnet50(pretrained=pretrained)
        elif (num == 101):
            self.net = tv.resnet101(pretrained=pretrained)
        elif (num == 152):
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h

        outputs = namedtuple("Outputs", ['relu1', 'conv2', 'conv3', 'conv4', 'conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)

        return out


class TransitionWithSkip(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, X):
        for module in self.module:
            X = module(X)
            if isinstance(module, torch.nn.ReLU):
                skip = X
        return X, skip


class densenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, num=121):
        super(densenet, self).__init__()
        if num == 121:
            pretrained_features = tv.densenet121(pretrained=pretrained).features
        elif num == 161:
            pretrained_features = tv.densenet161(pretrained=pretrained).features
        elif num == 169:
            pretrained_features = tv.densenet169(pretrained=pretrained).features
        elif num == 201:
            pretrained_features = tv.densenet201(pretrained=pretrained).features
        self.N_slices = 5

        self.slice1 = torch.nn.Sequential(pretrained_features.conv0,
                                          pretrained_features.norm0,
                                          pretrained_features.relu0
                                          )
        self.slice2 = torch.nn.Sequential(pretrained_features.pool0,
                                          pretrained_features.denseblock1,
                                          TransitionWithSkip(pretrained_features.transition1)
                                          )
        self.slice3 = torch.nn.Sequential(pretrained_features.denseblock2,
                                          TransitionWithSkip(pretrained_features.transition2)
                                          )
        self.slice4 = torch.nn.Sequential(pretrained_features.denseblock3,
                                          TransitionWithSkip(pretrained_features.transition3)
                                          )
        self.slice5 = torch.nn.Sequential(pretrained_features.denseblock4,
                                          pretrained_features.norm5)
        self.slices = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]

    def forward(self, X):
        features = []
        for i in range(self.N_slices):
            X = self.slices[i](X)
            if isinstance(X, (list, tuple)):
                X, skip = X
                features.append(skip)
            else:
                features.append(X)

        outputs = namedtuple("Outputs", ['slice1', 'slice2', 'slice3', 'slice4', 'slice5'])
        out = outputs(features[0], features[1], features[2], features[3], features[4])

        return out


class inception(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(inception, self).__init__()

        self.model = inceptionv4()
        self.slices = [self.model.features[:3],
                       self.model.features[3:5],
                       self.model.features[5:9],
                       self.model.features[9:15],
                       self.model.features[15:]]

    def forward(self, X):
        features = []
        for i in range(len(self.slices)):
            X = self.slices[i](X)
            features.append(X)

        outputs = namedtuple("Outputs", ['slice1', 'slice2', 'slice3', 'slice4', 'slice5'])
        out = outputs(features[0], features[1], features[2], features[3], features[4])

        return out


class mobilenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(mobilenet, self).__init__()

        pretrained_features = tv.mobilenet_v2(pretrained=pretrained).features
        self.slice1 = pretrained_features[:2]
        self.slice2 = pretrained_features[2:4]
        self.slice3 = pretrained_features[4:7]
        self.slice4 = pretrained_features[7:14]
        self.slice5 = pretrained_features[14:]
        self.slices = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]

    def forward(self, X):
        features = []
        for i in range(len(self.slices)):
            X = self.slices[i](X)
            features.append(X)

        outputs = namedtuple("Outputs", ['slice1', 'slice2', 'slice3', 'slice4', 'slice5'])
        out = outputs(features[0], features[1], features[2], features[3], features[4])

        return out


class xception(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(xception, self).__init__()

    def forward(self, X):
        return None


class senet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(senet, self).__init__()

    def forward(self, X):
        return None
