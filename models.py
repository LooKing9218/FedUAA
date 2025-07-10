import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict

import torchvision.models


class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
            
                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # print("x.shape ====== {}".format(x.shape))
        x = torch.flatten(x, 1)
        # print("x.shape ====== {}".format(x.shape))
        x = self.classifier(x)
        return x


class ResNetBackbone(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes):
        super(ResNetBackbone, self).__init__()
        print("=============== FC2 ===============")
        resnet = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-34
        resnet_modules = list(resnet.children())[:-2]
        self.features = nn.Sequential(*resnet_modules)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(2048, 512)),
                ('bn6', nn.BatchNorm1d(512)),
                ('relu6', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(512, num_classes)),
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
class ResNetBackboneGenPer(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes):
        super(ResNetBackboneGenPer, self).__init__()
        print("=============== ResNetBackboneGenPer ===============")
        resnet = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-34
        resnet_modules = list(resnet.children())[:-2]
        self.features = nn.Sequential(*resnet_modules)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(2048, 512))
            ])
        )
        self.classifier_Gen = nn.Sequential(
            OrderedDict([
                ('bnGen', nn.BatchNorm1d(512)),
                ('reluGen', nn.ReLU(inplace=True)),
                ('fc2Gen', nn.Linear(512, 2)),
            ])
        )
        self.classifier_Per = nn.Sequential(
            OrderedDict([
                ('bn6Per', nn.BatchNorm1d(512)),
                ('relu6Per', nn.ReLU(inplace=True)),
                ('fc2Per', nn.Linear(512, num_classes)),
            ])
        )



    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # print("x.shape ====== {}".format(x.shape))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x_gen = self.classifier_Gen(x)
        x_per = self.classifier_Per(x)
        return x_gen,x_per

class ResNetBackboneMoon(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes):
        super(ResNetBackboneMoon, self).__init__()
        print("=============== FC ===============")
        resnet = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-34
        resnet_modules = list(resnet.children())[:-2]
        self.features = nn.Sequential(*resnet_modules)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.FC1 = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(2048, 512)),
                ('bn6', nn.BatchNorm1d(512)),
                ('relu6', nn.ReLU(inplace=True)),
            ])
        )
        self.FC2 = nn.Sequential(
            OrderedDict([
                ('fc2', nn.Linear(512, num_classes)),
            ])
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # print("x.shape ====== {}".format(x.shape))
        x = torch.flatten(x, 1)
        x_fc1 = self.FC1(x)
        x_out = self.FC2(x_fc1)
        return x,x,x_out


class AlexNetCNN(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self):
        super(AlexNetCNN, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 512, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(512)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(512)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))



    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class ResNetCNN(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self):
        super(ResNetCNN, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-34
        resnet_modules = list(resnet.children())[:-2]
        self.features = nn.Sequential(*resnet_modules)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # print("x.shape ====== {}".format(x.shape))
        x = torch.flatten(x, 1)
        return x



class DisentanlgedFusion(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes):
        super(DisentanlgedFusion, self).__init__()
        self.BackboneGeneral = AlexNetCNN()
        self.BackboneSpecific = ResNetCNN()

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(4096, 1024)),
                ('bn6', nn.BatchNorm1d(1024)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(1024, 512)),
                ('bn7', nn.BatchNorm1d(512)),
                ('relu7', nn.ReLU(inplace=True)),

                ('fc3', nn.Linear(512, num_classes)),
            ])
        )

    def forward(self, x):
        x_gen = self.BackboneGeneral(x)
        x_sp = self.BackboneSpecific(x)
        x = torch.cat([x_gen,x_sp],dim=1)
        x = self.classifier(x)
        return x






class SAModuleBack(nn.Module):
    def __init__(self, n_dims, width=32, height=32):
        super(SAModuleBack, self).__init__()
        print("================= SAModuleBack =================")

        self.chanel_in = n_dims
        self.rel_h = nn.Parameter(torch.randn([1, n_dims, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims, 1, width]), requires_grad=True)

        self.query_conv = nn.Conv2d(
            in_channels = n_dims , out_channels = n_dims , kernel_size= 1)
        self.key_conv = nn.Conv2d(
            in_channels = n_dims , out_channels = n_dims , kernel_size= 1)
        self.value_conv = nn.Conv2d(
            in_channels = n_dims , out_channels = n_dims , kernel_size= 1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy_content = torch.bmm(proj_query, proj_key)


        content_position = (self.rel_h + self.rel_w).view(1, self.chanel_in, -1)
        content_position = torch.matmul(proj_query,content_position)
        energy = energy_content + content_position
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class SAModule(nn.Module):
    def __init__(self, n_dims, width=32, height=32):
        super(SAModule, self).__init__()
        print("================= SAModule =================")

        self.chanel_in = n_dims

        self.query_conv = nn.Conv2d(
            in_channels = n_dims , out_channels = n_dims , kernel_size= 1)
        self.key_conv = nn.Conv2d(
            in_channels = n_dims , out_channels = n_dims , kernel_size= 1)
        self.value_conv = nn.Conv2d(
            in_channels = n_dims , out_channels = n_dims , kernel_size= 1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out



class ResNetAttPrompt(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes):
        super(ResNetAttPrompt, self).__init__()
        print("=============== ResNetAttPrompt FC2 ===============")
        resnet = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-34
        self.conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )



        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.SA0 = SAModule(
            n_dims=64,width=64,height=64
        )

        self.SA1 = SAModule(
            n_dims=256,width=64,height=64
        )

        self.SA2 = SAModule(
            n_dims=512,width=32,height=32
        )

        self.SA3 = SAModule(
            n_dims=1024,width=16,height=16
        )

        self.SA4 = SAModule(
            n_dims=2048,width=8,height=8
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(2048, 512)),
                ('bn6', nn.BatchNorm1d(512)),
                ('relu6', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(512, num_classes)),
            ])
        )

    def Freeze(self):

        self.conv.eval()
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        self.layer4.eval()

    def forward(self, x, Freeze=True):
        x0 = self.conv(x)
        x_SA0 = self.SA0(x0)
        x1 = self.layer1(x_SA0)
        x_SA1 = self.SA1(x1)
        x2 = self.layer2(x_SA1)
        x_SA2 = self.SA2(x2)
        x3 = self.layer3(x_SA2)
        x_SA3 = self.SA3(x3)
        x4 = self.layer4(x_SA3)
        x_SA4 = self.SA4(x4)

        x = self.avgpool(x_SA4)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


