## Introduction.

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
        được thiết kế để mở rộng số lượng channel cho feature map.
    
    ConvBlock Class:
        
        • Mục đích: Khối convolution cơ bản để trích xuất đặc trưng
        • Thành phần: Hai nhóm Conv-BatchNorm-LeakyReLU liên tục. Khối này là một khối cơ
        bản trong U-Net, được sử dụng cho cả down-sampling và up-sampling.
    
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
        xuống bằn số lượng channel của ảnh màu.
    
    Unet Class:
    
        • Mục đích: Kết hợp tất cả các thành phần trên thành một kiến trúc U-Net đầy đủ.
        • Thành phần: Xử lý ảnh đầu vào bằng FirstFeature và ConvBlock. Bốn lớp Encoder với số
        channel tăng dần, mỗi lớp tiếp tục downsample và xử lý feature map. Bốn lớp Decoder với số
        channel giảm dần, mỗi lớp tăng kích thước, kết hợp đặc trưng từ encoder (skip connection).
    
    Một lớp FinalOutput để tạo ra ảnh đã được xử lý.
        
        • Forward: Đầu vào được xử lý qua các convolution ban đầu. Sau đó được downsample 4 lần,
        rồi được up sample lên 4 lần mỗi lần kết hợp với feature từ encoder. Cuối cùng đi qua lớp
        convolution cuối cùng để tạp ảnh đã xử lý.
    
##I. Task Super Resolution

    1. Yêu cầu Model: mạng Unet sử dụng Skip Connection và không có Skip Connection, sau đó so sánh kết quả giữa 2 model.
    2. Các bước thực hiện: 
        • Xây dựng dataset từ ảnh gốc (256x256x3). Khi load ảnh, mỗi sample cần có 2 ảnh input và target. Input là: ảnh gốc resize 4 lần (64x64x3), target là ảnh gốc (256x256x3).
        • Chia data thành các tập train, validation.
        • Normalize data phải phù hợp với activation của layer cuối trong model để đảm bảo output có giá trị             trong range các giá trị của ảnh thông thường.
        • Lựa chọn Loss và metrics phù hợp cho bài toán.
        • Config các hyperparameter.
        • Train và test kết quả.
        • Ta sẽ xây dựng từng file .py bao gồm: 
            - Library.py (file chứ các thư viện cần thiết).
            - Prepare_Dataset.py (file chuẩn bị dataset cho model).
            - Train_Val.py (file gồm các hàm được dùng để huấn luyện và đánh giá kết quả)
            - UnetNoSkipConnection.py (model Unet không sử dụng SkipConnection).
            - UnetSkipConnection.py (model Unet sử dụng SkipConnection).
            - 2 file ipynb để train 2 model trên sử dụng gpu của colab.

    3. Kết quả của sau khi train.
    
        Đối với model Unet sử dụng Skip Connection:
        
        
![image](https://github.com/PhamTrinhDuc/Unet-Architecture/assets/127647215/8f614901-dd5e-4e58-a967-2c89e24f0ac0)


![Screenshot 2024-02-19 203758](https://github.com/PhamTrinhDuc/Unet-Architecture/assets/127647215/6968020c-3ed0-4fbb-beb0-569675bf3595)


        Đối với model Unet không sử dụng Skip Connection:
        

![image](https://github.com/PhamTrinhDuc/Unet-Architecture/assets/127647215/59c3c58d-6aaa-4b9b-9819-3ce29874192e)


![Screenshot 2024-02-19 204058](https://github.com/PhamTrinhDuc/Unet-Architecture/assets/127647215/50abe2a5-de80-48dd-bbc4-ed9403ce673b)


    4. Nhận xét: Xem xét 2 hình trên ta thấy được mạng có sử dụng Skip Connection tốt hơn nhiều so
    
với mạng không sử dụng Skip Connection. Một số chức năng của Skip Connection: 
    
        • Kết hợp các đặc trưng từ nhiều cấp độ khác nhau của ảnh, giúp ảnh có độ chính xác cao hơn.
        
        • Giảm thiếu vấn đề Gradient Vanishing.
        
        • Cải thiện độ chính xác.
        
        • Giảm nguy cơ overfitting.
        

##II. Image Painting.
    1. Yêu cầu Model: mạng Unet sử dụng Skip Connection.
    2. Các bước thực hiện: 
        • Xây dựng datset từ ảnh gốc. Khi load ảnh mỗi sample các bạn tạo ra 2 ảnh (input và target).
Input là ảnh gốc và được vẽ thêm các line random, target là ảnh gốc.
        • Chia data thành các tập train, validation.
        • Normalize data phải phù hợp với activation của layer cuối trong model để đảm bảo output có giá trị             trong range các giá trị của ảnh thông thường.
        • Lựa chọn Loss và metrics phù hợp cho bài toán.
        • Config các hyperparameter.
        • Train và test kết quả.
        • Ta sẽ xây dựng từng file .py bao gồm: 
            - Library.py (file chứ các thư viện cần thiết).
            - Prepare_Dataset.py (file chuẩn bị dataset cho model).
            - Train_Val.py (file gồm các hàm được dùng để huấn luyện và đánh giá kết quả)
            - UnetSkipConnection.py (file model Unet sử dụng SkipConnection).
            - run_modelPainting.ipynb (file train model sử dụng gpu).
    
    3. Kết quả của sau khi train.

![image](https://github.com/PhamTrinhDuc/Unet-Architecture/assets/127647215/539bb2e6-72f9-4334-8bf5-7ee5f8f686d3)


![Screenshot 2024-02-19 212917](https://github.com/PhamTrinhDuc/Unet-Architecture/assets/127647215/c7b80019-04fa-4160-9528-e4b27acfd124)


![Screenshot 2024-02-19 212937](https://github.com/PhamTrinhDuc/Unet-Architecture/assets/127647215/87eea181-c054-4749-9fd9-92ddc9460874)

    4. Nhận xét:

    Đối với những line lớn model cho kết quả chưa được tốt lắm, ngược lại với các line nhỏ thì kết quả đã khá tốt.
        
