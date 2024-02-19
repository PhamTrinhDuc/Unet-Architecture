## Introduction

1. Giới thiệu mạng Unet: 
    UNET là một mô hình mạng nơ-ron sâu được phát triển bởi Olaf Ronneberger, Philipp Fischer
    và Thomas Brox vào năm 2015, chủ yếu được sử dụng trong lĩnh vực xử lý ảnh và trí tuệ nhân
    tạo (AI). Mô hình này đã đạt được sự phổ biến rộng rãi trong các ứng dụng liên quan đến phân
    đoạn ảnh, như phân đoạn tế bào trong hình ảnh y học hoặc phân đoạn đối tượng trong hình ảnh
    chất lượng cao.
    
    UNET sử dụng kiến trúc mạng Encoder-Decoder, với mục đích chính của lớp skip connection là
    tạo đường dẫn ngắn từ đầu vào đến đầu ra. Hàm kích hoạt thường được sử dụng trong UNET là
    Rectified Linear Unit (ReLU).

2. Ở task này ta sẽ làm 2 ứng dụng của mạng Unet là: tăng độ phân giải của ảnh(Super Resolution) và tái tạo một phần bị thiếu hay bị che khuất của ảnh(Image Painting).

3. Cấu trúc mạng Unet trong task này: 

FirstFeature Class:
    • Mục đích: Tạo Tạo feature map ban đầu từ input
    • Thành phần: Một lớp convolution duy nhất theo sau là hàm LeakyReLU. convolution này
    sử dụng kích thước kernel là 1, bước nhảy là 1 và không padding. Đây là một lớp đơn giản
    được thiết kế để mở rộng số lượng channel cho feature map

ConvBlock Class:
    
    • Mục đích: Khối convolution cơ bản để trích xuất đặc trưng
    • Thành phần: Hai nhóm Conv-BatchNorm-LeakyReLU liên tục. Khối này là một khối cơ
    bản trong U-Net, được sử dụng cho cả down-sampling và up-sampling

Encoder Class:
    
    • Mục đích: Để giảm size của feature map và trích xuất các high-level feature.
    • Thành phần: Một lớp Max Pooling theo sau là ConvBlock. Max Pooling giảm kích thước
    xuống một nửa, trong khi ConvBlock xử lý các đặc trưng.

Decoder Class:

    • Mục đích: Để tăng kích thước feature map và kết hợp với feature map tương ứng từ Encoder
    (skip connection).
    • Thành phần: Upsampling (sử dụng nội suy bilinear) để tăng kích thước không gian. Một
    lớp convolution để giảm số lượng channel. Một ConvBlock để xử lý các feature được ghép
    (từ lớp upsampling và skip connection).

FinalOutput Class:

    • Mục đích: Tạo ra đầu ra cuối cùng từ feature map cuối cùng.
    • Thành phần: Một lớp convolution với hàm Tanh. Điều này giảm số lượng channel đầu ra
    xuống bằn số lượng channel của ảnh màu

Unet Class:

    • Mục đích: Kết hợp tất cả các thành phần trên thành một kiến trúc U-Net đầy đủ.
    • Thành phần: Xử lý ảnh đầu vào bằng FirstFeature và ConvBlock. Bốn lớp Encoder với số
    channel tăng dần, mỗi lớp tiếp tục downsample và xử lý feature map. Bốn lớp Decoder với số
    channel giảm dần, mỗi lớp tăng kích thước, kết hợp đặc trưng từ encoder (skip connection).

Một lớp FinalOutput để tạo ra ảnh đã được xử lý.
    
    • Forward: Đầu vào được xử lý qua các convolution ban đầu. Sau đó được downsample 4 lần,
    rồi được up sample lên 4 lần mỗi lần kết hợp với feature từ encoder. Cuối cùng đi qua lớp
    convolution cuối cùng để tạp ảnh đã xử lý

## 1. Task Super Resolution


