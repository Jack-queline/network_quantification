import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *

class NetBN(nn.Module):

    def __init__(self, num_channels=3):
        super(NetBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(128, 256, 3, 2, 1)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256, 512, 3, 2, 1)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn16 = nn.BatchNorm2d(512)
        self.conv17 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn17 = nn.BatchNorm2d(512)
        self.shortcut1 = nn.Conv2d(64,128,1,2)
        self.shortcut2 = nn.Conv2d(128,256,1,2)
        self.shortcut3 = nn.Conv2d(256,512,1,2)
        # 全连接层
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        x = F.relu(self.bn1(out))

        out = self.conv2(x)
        out = F.relu(self.bn2(out))
        out = self.conv3(out)
        out = self.bn3(out)
        x = F.relu(out+x)
        out = self.conv4(x)
        out = F.relu(self.bn4(out))
        out = self.conv5(out)
        out = self.bn5(out)
        x =F.relu(out+x)


        out = self.conv6(x)
        out = F.relu(self.bn6(out))
        out = self.conv7(out)
        out = self.bn7(out)
        shortcut = self.shortcut1(x)
        shortcut = self.bn7(shortcut)
        x = F.relu(out+shortcut)
        out = self.conv8(x)
        out = F.relu(self.bn8(out))
        out = self.conv9(out)
        out = self.bn9(out)
        x =F.relu(out+x)
        
        out = self.conv10(x)
        out = F.relu(self.bn10(out))
        out = self.conv11(out)
        out = self.bn11(out)
        shortcut = self.shortcut2(x)
        shortcut = self.bn11(shortcut)
        x = F.relu(out+shortcut)
        out = self.conv12(x)
        out = F.relu(self.bn12(out))
        out = self.conv13(out)
        out = self.bn13(out)
        x =F.relu(out+x)

        out = self.conv14(x)
        out = F.relu(self.bn14(out))
        out = self.conv15(out)
        out = self.bn15(out)
        shortcut = self.shortcut3(x)
        shortcut = self.bn15(shortcut)
        x = F.relu(out+shortcut)
        out = self.conv16(x)
        out = F.relu(self.bn16(out))
        out = self.conv17(out)
        out = self.bn17(out)
        x =F.relu(out+x)
        
        x = F.avg_pool2d(x, 4)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def quantize(self, num_bits=8):
        self.qconv1 = QConvBNReLU(self.conv1, self.bn1, qi=True, qo=True, num_bits=num_bits)
        self.qconv2 = QConvBNReLU(self.conv2, self.bn2, qi=False, qo=True, num_bits=num_bits)
        self.qconv3 = QConvBN(self.conv3, self.bn3, qi=False, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU()
        self.qconv4 = QConvBNReLU(self.conv4, self.bn4, qi=False, qo=True, num_bits=num_bits)
        self.qconv5 = QConvBN(self.conv5, self.bn5, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU()
        self.qconv6 = QConvBNReLU(self.conv6, self.bn6, qi=False, qo=True, num_bits=num_bits)
        self.qconv7 = QConvBN(self.conv7, self.bn7, qi=False, qo=True, num_bits=num_bits)
        self.qshortcut1 = QConvBN(self.shortcut1, self.bn7, qi=False, qo=True, num_bits=num_bits)
        self.qrelu3 = QReLU()
        self.qconv8 = QConvBNReLU(self.conv8, self.bn8, qi=False, qo=True, num_bits=num_bits)
        self.qconv9 = QConvBN(self.conv9, self.bn9, qi=False, qo=True, num_bits=num_bits)
        self.qrelu4 = QReLU()
        self.qconv10 = QConvBNReLU(self.conv10, self.bn10, qi=False, qo=True, num_bits=num_bits)
        self.qconv11 = QConvBN(self.conv11, self.bn11, qi=False, qo=True, num_bits=num_bits)
        self.qshortcut2 = QConvBN(self.shortcut2, self.bn11, qi=False, qo=True, num_bits=num_bits)
        self.qrelu5 = QReLU()
        self.qconv12 = QConvBNReLU(self.conv12, self.bn12, qi=False, qo=True, num_bits=num_bits)
        self.qconv13 = QConvBN(self.conv13, self.bn13, qi=False, qo=True, num_bits=num_bits)
        self.qrelu6 = QReLU()
        self.qconv14 = QConvBNReLU(self.conv14, self.bn14, qi=False, qo=True, num_bits=num_bits)
        self.qconv15 = QConvBN(self.conv15, self.bn15, qi=False, qo=True, num_bits=num_bits)
        self.qshortcut3 = QConvBN(self.shortcut3, self.bn15, qi=False, qo=True, num_bits=num_bits)
        self.qrelu7 = QReLU()
        self.qconv16 = QConvBNReLU(self.conv16, self.bn16, qi=False, qo=True, num_bits=num_bits)
        self.qconv17 = QConvBN(self.conv17, self.bn17, qi=False, qo=True, num_bits=num_bits)
        self.qrelu8 = QReLU()
        self.qavgpool2d_1 = QAvgPooling2d(kernel_size=4, stride=None, padding=0)
        self.qfc1 = QLinear(self.fc1, qi=False, qo=True, num_bits=num_bits)
       

    def quantize_forward(self, x):
        x = self.qconv1(x)

        out = self.qconv2(x)
        out = self.qconv3(out)
        x = self.qrelu1(out+x)
        out = self.qconv4(x)
        out = self.qconv5(out)
        x = self.qrelu2(out+x)
        
        out = self.qconv6(x)
        out = self.qconv7(out)
        shortcut = self.qshortcut1(x)
        x = self.qrelu3(out+shortcut)
        out = self.qconv8(x)
        out = self.qconv9(x)
        x = self.qrelu4(out+x)

        out = self.qconv10(x)
        out = self.qconv11(out)
        shortcut = self.qshortcut2(x)
        x = self.qrelu5(out+shortcut)
        out = self.qconv12(x)
        out = self.qconv13(x)
        x = self.qrelu6(out+x)

        out = self.qconv14(x)
        out = self.qconv15(out)
        shortcut = self.qshortcut3(x)
        x = self.qrelu7(out+shortcut)
        out = self.qconv16(x)
        out = self.qconv17(x)
        x = self.qrelu8(out+x)
        
        x = self.qavgpool2d_1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.qfc1(x)
        return x

    def freeze(self):
        self.qconv1.freeze()

        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qconv3.freeze(qi=self.qconv2.qo)
        self.qconv2.qo.scale = self.qconv1.qo.scale + self.qconv3.qo.scale
        self.qconv2.qo.zero_point = self.qconv1.qo.zero_point + self.qconv3.qo.zero_point
        self.qconv2.qo.min = self.qconv1.qo.min + self.qconv3.qo.min
        self.qconv2.qo.max = self.qconv1.qo.max + self.qconv3.qo.max
        self.qrelu1.freeze(self.qconv2.qo)
        self.qconv4.freeze(qi=self.qconv2.qo)
        self.qconv5.freeze(qi=self.qconv4.qo)
        self.qconv4.qo.scale = self.qconv2.qo.scale + self.qconv5.qo.scale
        self.qconv4.qo.zero_point = self.qconv2.qo.zero_point + self.qconv5.qo.zero_point
        self.qconv4.qo.min = self.qconv2.qo.min + self.qconv5.qo.min
        self.qconv4.qo.max = self.qconv2.qo.max + self.qconv5.qo.max
        self.qrelu2.freeze(self.qconv4.qo)

        self.qconv6.freeze(qi=self.qconv4.qo)
        self.qconv7.freeze(qi=self.qconv6.qo)
        self.qshortcut1.freeze(qi=self.qconv4.qo)
        self.qconv6.qo.scale = self.qshortcut1.qo.scale + self.qconv7.qo.scale
        self.qconv6.qo.zero_point = self.qshortcut1.qo.zero_point + self.qconv7.qo.zero_point
        self.qconv6.qo.min = self.qshortcut1.qo.min + self.qconv7.qo.min
        self.qconv6.qo.max = self.qshortcut1.qo.max + self.qconv7.qo.max
        self.qrelu3.freeze(self.qconv6.qo)
        self.qconv8.freeze(qi=self.qconv6.qo)
        self.qconv9.freeze(qi=self.qconv8.qo)
        self.qconv8.qo.scale = self.qconv6.qo.scale + self.qconv9.qo.scale
        self.qconv8.qo.zero_point = self.qconv6.qo.zero_point + self.qconv9.qo.zero_point
        self.qconv8.qo.min = self.qconv6.qo.min + self.qconv9.qo.min
        self.qconv8.qo.max = self.qconv6.qo.max + self.qconv9.qo.max
        self.qrelu4.freeze(self.qconv8.qo)

        self.qconv10.freeze(qi=self.qconv8.qo)
        self.qconv11.freeze(qi=self.qconv10.qo)
        self.qshortcut2.freeze(qi=self.qconv8.qo)
        self.qconv10.qo.scale = self.qshortcut2.qo.scale + self.qconv11.qo.scale
        self.qconv10.qo.zero_point = self.qshortcut2.qo.zero_point + self.qconv11.qo.zero_point
        self.qconv10.qo.min = self.qshortcut2.qo.min + self.qconv11.qo.min
        self.qconv10.qo.max = self.qshortcut2.qo.max + self.qconv11.qo.max
        self.qrelu5.freeze(self.qconv10.qo)
        self.qconv12.freeze(qi=self.qconv10.qo)
        self.qconv13.freeze(qi=self.qconv12.qo)
        self.qconv12.qo.scale = self.qconv10.qo.scale + self.qconv13.qo.scale
        self.qconv12.qo.zero_point = self.qconv10.qo.zero_point + self.qconv13.qo.zero_point
        self.qconv12.qo.min = self.qconv10.qo.min + self.qconv13.qo.min
        self.qconv12.qo.max = self.qconv10.qo.max + self.qconv13.qo.max
        self.qrelu6.freeze(self.qconv12.qo)

        self.qconv14.freeze(qi=self.qconv12.qo)
        self.qconv15.freeze(qi=self.qconv14.qo)
        self.qshortcut3.freeze(qi=self.qconv12.qo)
        self.qconv14.qo.scale = self.qshortcut3.qo.scale + self.qconv15.qo.scale
        self.qconv14.qo.zero_point = self.qshortcut3.qo.zero_point + self.qconv15.qo.zero_point
        self.qconv14.qo.min = self.qshortcut3.qo.min + self.qconv15.qo.min
        self.qconv14.qo.max = self.qshortcut3.qo.max + self.qconv15.qo.max
        self.qrelu7.freeze(self.qconv14.qo)
        self.qconv16.freeze(qi=self.qconv14.qo)
        self.qconv17.freeze(qi=self.qconv16.qo)
        self.qconv16.qo.scale = self.qconv14.qo.scale + self.qconv17.qo.scale
        self.qconv16.qo.zero_point = self.qconv14.qo.zero_point + self.qconv17.qo.zero_point
        self.qconv16.qo.min = self.qconv14.qo.min + self.qconv17.qo.min
        self.qconv16.qo.max = self.qconv14.qo.max + self.qconv17.qo.max
        self.qrelu8.freeze(self.qconv16.qo)

        self.qavgpool2d_1.freeze(self.qconv16.qo)
        self.qfc1.freeze(qi=self.qconv16.qo)

        
    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)

        out = self.qconv2.quantize_inference(qx)
        out = self.qconv3.quantize_inference(out)
        qx = self.qrelu1.quantize_inference(out+qx)
        out = self.qconv4.quantize_inference(qx)
        out = self.qconv5.quantize_inference(out)
        qx = self.qrelu2.quantize_inference(out+qx)

        out = self.qconv6.quantize_inference(qx)
        out = self.qconv7.quantize_inference(out)
        shortcut = self.qshortcut1.quantize_inference(qx)
        qx = self.qrelu3.quantize_inference(out+shortcut)
        out = self.qconv8.quantize_inference(qx)
        out = self.qconv9.quantize_inference(out)
        qx = self.qrelu4.quantize_inference(out+qx)

        out = self.qconv10.quantize_inference(qx)
        out = self.qconv11.quantize_inference(out)
        shortcut = self.qshortcut2.quantize_inference(qx)
        qx = self.qrelu5.quantize_inference(out+shortcut)
        out = self.qconv12.quantize_inference(qx)
        out = self.qconv13.quantize_inference(out)
        qx = self.qrelu6.quantize_inference(out+qx)

        out = self.qconv14.quantize_inference(qx)
        out = self.qconv15.quantize_inference(out)
        shortcut = self.qshortcut3.quantize_inference(qx)
        qx = self.qrelu7.quantize_inference(out+shortcut)
        out = self.qconv16.quantize_inference(qx)
        out = self.qconv17.quantize_inference(out)
        qx = self.qrelu8.quantize_inference(out+qx)

        qx = self.qavgpool2d_1.quantize_inference(qx)
        qx = qx.reshape(qx.shape[0], -1)
        qx = self.qfc1.quantize_inference(qx)
        out = self.qfc1.qo.dequantize_tensor(qx)
        return out