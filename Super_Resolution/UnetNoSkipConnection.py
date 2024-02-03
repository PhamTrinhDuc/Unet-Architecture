from Unet_Architecture.Super_Resolution.Library import *


class FirstFeatureNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstFeatureNoSkip, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlockNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockNoSkip, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderNoSkip, self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlockNoSkip(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class DecoderNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderNoSkip, self).__init__()
        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            # out_channels * 2: để cân bằng weight với khi dùng skip connection
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_block = ConvBlockNoSkip(out_channels * 2, out_channels)

    def forward(self, x):
        x = self.decoder(x)
        x = self.conv_block(x)
        return x


class FinalFeatureNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalFeatureNoSkip, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.block(x)


class UnetNoSkip(nn.Module):
    def __init__(self, in_channels, num_classes, img_height, img_width):
        super(UnetNoSkip, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # do ảnh input đã giảm đi 4 lần, resize lên để thực hiện training, sau khi có kết quả sẽ so sánh với ảnh ban đầu chưa được resize
        self.resize_trueForm = transforms.Resize((img_height * 4, img_width * 4), antialias=True)
        self.conv1 = ConvBlockNoSkip(in_channels, 64)
        self.conv2 = ConvBlockNoSkip(64, 64)

        self.encoder1 = EncoderNoSkip(64, 128)
        self.encoder2 = EncoderNoSkip(128, 256)
        self.encoder3 = EncoderNoSkip(256, 512)
        self.encoder4 = EncoderNoSkip(512, 1024)

        self.decoder1 = DecoderNoSkip(1024, 512)
        self.decoder2 = DecoderNoSkip(512, 256)
        self.decoder3 = DecoderNoSkip(256, 128)
        self.decoder4 = DecoderNoSkip(128, 64)

        self.out_conv = FinalFeatureNoSkip(64, num_classes)

    def forward(self, x):
        x = self.resize_trueForm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.out_conv(x)
        return x


if __name__ == "__main__":
    in_channels, num_classes = 3, 3
    model = UnetNoSkip(in_channels, num_classes, 64, 64)
    # summary(model, (3, 128, 128))
    input = torch.ones(2, 3, 64, 64)
    output = model(input)
    print(output.shape)
