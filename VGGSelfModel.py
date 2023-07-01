import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
class MSCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSCM, self).__init__()

        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.score = nn.Conv2d(out_channel * 4, 256, 3, padding=1)
    def forward(self, x):
        x = self.convert(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.score(x)

        return x
class RA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.atten = nn.Conv2d(out_channel, 1, 3, padding=1)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.conv_e = nn.Conv2d(out_channel*2+64, out_channel//2, 3, padding=1)
        self.channel = out_channel
        self.paras = torch.nn.Parameter(torch.FloatTensor(1), requires_grad = True)
        self.paras.data.fill_(1)

    def forward(self, x, y, z):
        a = torch.sigmoid(-y) #channel = 256、128、64、32
        x = self.convert(x) #channel = 256、128、64、32
        x0 = self.atten(x) #channel = 1
        x0 = torch.sigmoid(x0) #channel = 1
        x = x0.expand(-1, self.channel, -1, -1).mul(x)+x #channel = 256、128、64、32
        x = a.mul(x) #channel = 256、128、64、32
        y = torch.cat((self.convs(x), y, (self.paras*z)), 1)
        y = self.conv_e(y)

        return y


class CA(nn.Module):
    def __init__(self, in_channel):
        super(CA, self).__init__()

        self.convert = nn.Conv2d(in_channel, 64, 1)
        self.query_conv = nn.Conv2d(64, 64, 1)
        self.key_conv = nn.Conv2d(64, 64, 1)
        self.value_conv = nn.Conv2d(64, 64, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)


        self.conv1 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.convert(x)
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # 转置，C*N-->N*C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)  # 矩阵相乘
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        x0 = x.mean(dim=(2, 3), keepdim=True)  # channel = 64
        x0 = F.relu(self.bn1(self.conv1(x0)), inplace=True)
        x0 = torch.sigmoid(self.bn2(self.conv2(x0)))

        out = self.gamma * out + x + x0 * x

        return out

class VGGSelfModel(nn.Module):

    def __init__(self):
        super(VGGSelfModel, self).__init__()
        self.vgg_layer0 = models.vgg16_bn(pretrained=True).features[:7]
        self.vgg_layer1 = models.vgg16_bn(pretrained=True).features[7:14]
        self.vgg_layer2 = models.vgg16_bn(pretrained=True).features[14:24]
        self.vgg_layer3 = models.vgg16_bn(pretrained=True).features[24:34]
        self.vgg_layer4 = models.vgg16_bn(pretrained=True).features[34:43]
        self.sig = nn.Sigmoid()
        self.conv_end = nn.Conv2d(16, 1, 3, padding=1)


        #测试
        self.mscm = MSCM(512, 256)

        self.ca1 = CA(256)
        self.ca2 = CA(256)
        self.ca3 = CA(256)
        self.ca4 = CA(256)

        self.ra1 = RA(64, 32)
        self.ra2 = RA(128, 64)
        self.ra3 = RA(256, 128)
        self.ra4 = RA(512, 256)

        #测试

    def forward(self, x):
        x1 = self.vgg_layer0(x)
        x2 = self.vgg_layer1(x1)
        x3 = self.vgg_layer2(x2)
        x4 = self.vgg_layer3(x3)
        x5 = self.vgg_layer4(x4)  # 32

        #测试
        x_size = x.size()[2:]
        x1_size = x1.size()[2:]
        x2_size = x2.size()[2:]
        x3_size = x3.size()[2:]
        x4_size = x4.size()[2:]

        y5 = self.mscm(x5)

        z5 = self.ca4(y5)
        z4 = self.ca3(y5)
        z3 = self.ca2(y5)
        z2 = self.ca1(y5)


        y5_4 = F.interpolate(y5, x4_size, mode='bilinear', align_corners=True)#上采样一次，未sigmoid
        z5_4 = F.interpolate(z5, x4_size, mode='bilinear', align_corners=True)
        y4 = self.ra4(x4, y5_4, z5_4)#先减再sigmoid


        y4_3 = F.interpolate(y4, x3_size, mode='bilinear', align_corners=True)#y4已经包含了第4、5阶的和
        z5_3 = F.interpolate(z4, x3_size, mode='bilinear', align_corners=True)
        y3 = self.ra3(x3, y4_3, z5_3)


        y3_2 = F.interpolate(y3, x2_size, mode='bilinear', align_corners=True)
        z5_2 = F.interpolate(z3, x2_size, mode='bilinear', align_corners=True)
        y2 = self.ra2(x2, y3_2, z5_2)


        y2_1 = F.interpolate(y2, x1_size, mode='bilinear', align_corners=True)
        z5_1 = F.interpolate(z2, x1_size, mode='bilinear', align_corners=True)
        y1 = self.ra1(x1, y2_1, z5_1)


        y1 = self.conv_end(y1)
        s1 = self.sig(y1)
        score1 = F.interpolate(s1, x_size, mode='bilinear', align_corners=True)

        score1 = score1.squeeze(1)
        return score1

