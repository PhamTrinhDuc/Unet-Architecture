from Unet_Architecture.Image_Painting.Library import *


class ImageDataset(Dataset):
    def __init__(self, data_dir, img_width, img_height, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        self.img_width = img_width
        self.img_height = img_height
        self.images = os.listdir(self.data_dir)

    def normalize(self, input_img, target_img):
        # do pytorch đã scale ảnh về (0, 1), ta cần scale về (-1, 1)
        input_img = input_img * 2 - 1
        target_img = target_img * 2 - 1

        return input_img, target_img

    def random_transform(self, input_image, target_image):
        if torch.rand([]) < 0.5:
            input_image = transforms.functional.hflip(input_image)
            target_image = transforms.functional.hflip(target_image)
        return input_image, target_image

    def create_mask(self, image, img_height, img_width):
        mask_img = image.copy()
        mask = np.full((img_height, img_width, 3), 0, dtype=np.uint8)
        for _ in range(np.random.randint(1, 5)):
            # tạo tọa độ 2 điểm trên ảnh
            x1, y1 = np.random.randint(1, img_width), np.random.randint(1, img_height)
            x2, y2 = np.random.randint(1, img_width), np.random.randint(1, img_height)
            thick = np.random.randint(1, 15)

            cv2.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness=thick)
        # với các ptu mask có gt true, gt 255 sẽ được gán cho phần tử tương ứng trong mask_image
        # ngược các pt trong mask_image sẽ được giữ nguyên
        mask_image = np.where(mask, 255 * np.ones_like(mask), mask_img)
        return mask_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir, self.images[item])
        image = np.array(Image.open(img_path).convert("RGB"))

        input_image = self.create_mask(image, self.img_height, self.img_width)
        input_image = transforms.functional.to_tensor(input_image)
        target_image = transforms.functional.to_tensor(image)

        input_image, target_image = self.normalize(input_image, target_image)

        if self.is_train:
            input_image, target_image = self.random_transform(input_image, target_image)

        return input_image, target_image


def visualize_data(train_loader):
    input_batch, target_batch = next(iter(train_loader))
    # đưa ảnh về (0, 1) để visualize
    input_batch = (input_batch + 1) / 2
    target_batch = (target_batch + 1) / 2

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(2, 2, 1)
    plt.title("Input")
    plt.imshow(input_batch[0].numpy().transpose(1, 2, 0))
    plt.axis('off')

    ax = plt.subplot(2, 2, 2)
    plt.title("Target")
    plt.imshow(target_batch[0].numpy().transpose(1, 2, 0))
    plt.axis('off')

    ax = plt.subplot(2, 2, 3)
    plt.title("Input")
    plt.imshow(input_batch[1].numpy().transpose(1, 2, 0))
    plt.axis('off')

    ax = plt.subplot(2, 2, 4)
    plt.title("Target")
    plt.imshow(target_batch[1].numpy().transpose(1, 2, 0))
    plt.axis('off')
    plt.show()

def data(path_train, path_val, width_size, height_size, batch_size):
    train_dataset = ImageDataset(path_train, width_size, height_size, True)
    val_dataset = ImageDataset(path_val, width_size, height_size, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, val_dataset, train_loader, val_loader


if __name__ == "__main__":
    width_size = 256
    height_size = 256
    batch_size = 8
    path_train = "./dataset/dataset/train"
    path_val = "./dataset/dataset/val"
    # images = os.listdir(path_train)

    # train_dataset = ImageDataset(path_train, width_size, height_size, True)
    # val_dataset = ImageDataset(path_val, width_size, height_size, False)
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_dataset, val_dataset, train_loader, val_loader = data(path_train, path_val, width_size, height_size, batch_size)
    print(f"Number of image train: {len(train_dataset)} ||Number of image val: {len(val_dataset)}")
    print(f"Number of train batch: {len(train_loader)} || Number of val batch: {len(val_loader)}")

    visualize_data(train_loader)
