from Unet_Architecture.Super_Resolution.Library import *


class FirstFeatures(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstFeatures, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
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


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.decoder(x)
        x = self.conv_block(torch.concat([x, skip], dim=1))
        return x


class FinalFeatures(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalFeatures, self).__init__()
        self.end_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.end_conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels, num_classes, img_height, img_width):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.resize_trueform = transforms.Resize((img_height * 4, img_width * 4), antialias=True)
        self.first_features = FirstFeatures(in_channels, 64)
        self.conv1 = ConvBlock(64, 64)
        self.encoder1 = Encoder(64, 128)
        self.encoder2 = Encoder(128, 256)
        self.encoder3 = Encoder(256, 512)
        self.encoder4 = Encoder(512, 1024)

        self.decoder1 = Decoder(1024, 512)
        self.decoder2 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder4 = Decoder(128, 64)
        self.end_features = FinalFeatures(64, num_classes)

    def forward(self, x):
        x = self.resize_trueform(x)
        x = self.first_features(x)
        x1 = self.conv1(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        x = self.decoder1(x5, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)
        x = self.end_features(x)
        return x


if __name__ == "__main__":
    model = Unet(3, 3, 64, 64)
    summary(model, (3, 256, 256))
    input = torch.ones(2, 3, 64, 64)
    output = model(input)
    print(output.shape)
