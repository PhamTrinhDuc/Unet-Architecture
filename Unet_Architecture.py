import torch
import torch.nn as nn
from torchsummary import summary


# Much đích: tạo feature map từ input
# Thành phần: tạo 1 lớp convolution duy nhất có kernel_size=1, stride=1, padding=0. Sau đó là activation func: LeakyReLU
# LeakyReLU: input được scale về (-1, 1), nếu dùng ReLU các giá trị âm sẽ thành 0 và làm mất thông tin,
# thay vào đó ta sử dụng LeakyReLU
class FirstFeature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstFeature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# Mục đích: trích xuất đặc trưng
# Thành phần: tạo hai nhóm Conv-BatchNorm-LeakyReLU,
# khối này là khối cơ bản trong Unet được sử dụng trong encoder và decoder
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# Mục đích: Để giảm size của feature map và trích xuất các high-level feature.
# Thành phần: Một lớp Max Pooling theo sau là ConvBlock. Max Pooling giảm kích thước
# xuống một nửa, trong khi ConvBlock xử lý các đặc trưng.

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# Mục đích: Để tăng kích thước feature map và kết hợp với feature map tương ứng từ Encoder
# (skip connection).
# Thành phần: Upsampling (sử dụng nội suy bilinear) để tăng kích thước không gian. Một
# lớp convolution để giảm số lượng channel. Một ConvBlock để xử lý các feature được ghép
# (từ lớp upsampling và skip connection).
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.decoder(x)
        x = torch.concat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


# Mục đích: Tạo ra đầu ra cuối cùng từ feature map cuối cùng.
# Thành phần: Một lớp convolution với hàm Tanh. Điều này giảm số lượng channel đầu ra
# xuống bằn số lượng channel của ảnh màu
class FinalOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()  # đưa output từ (-1, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, n_channels=3, num_classes=3):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes

        self.conv_start = FirstFeature(self.n_channels, 64)
        self.conv = ConvBlock(64, 64)
        self.encoder1 = Encoder(64, 128)
        self.encoder2 = Encoder(128, 256)
        self.encoder3 = Encoder(256, 512)
        self.encoder4 = Encoder(512, 1024)

        self.decoder1 = Decoder(1024, 512)
        self.decoder2 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder4 = Decoder(128, 64)

        self.conv_end = FinalOutput(64, self.num_classes)

    def forward(self, x):
        x = self.conv_start(x)
        x1 = self.conv(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        x = self.decoder1(x5, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)
        x = self.conv_end(x)
        return x


if __name__ == "__main__":
    model = Unet(3, 3)
    # summary(model, (3, 128, 128))
    input = torch.ones(2, 3, 128, 128)
    print(model(input).shape)
