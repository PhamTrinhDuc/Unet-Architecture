from Unet_Architecture.Super_Resolution.Library import *


def generate_images(model, inputs, labels):
    model.eval()
    with torch.no_grad():
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs)
    inputs, labels, predictions = inputs.cpu().numpy(), labels.cpu().numpy(), predictions.cpu().numpy()
    plt.figure(figsize=(15, 20))

    display_list = [inputs[-1].transpose((1, 2, 0)), labels[-1].transpose((1, 2, 0)),
                    predictions[-1].transpose((1, 2, 0))]
    title = ['Input', 'Real', 'Predicted']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow((display_list[i] + 1) / 2)
        plt.axis('off')
    plt.show()


def plot_result(num_epochs, train_psnrs, eval_psnrs, train_losses, eval_losses):
    epochs = list(range(num_epochs))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs[0].plot(epochs, train_psnrs, label="Training")
    axs[0].plot(epochs, eval_psnrs, label="Evaluation")
    axs[1].plot(epochs, train_losses, label="Training")
    axs[1].plot(epochs, eval_losses, label="Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[0].set_ylabel("PSNR")
    axs[1].set_ylabel("Loss")
    plt.legend()


def predict_and_display(model, test_dataloader, device):
    model.eval()

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            if idx >= 10:
                break
            inputs = inputs.to(device)
            predictions = model(inputs)
            generate_images(model, inputs, labels)
            plt.show()


def train_epoch(model, optimizer, criterion, train_loader, device, epoch=0, log_interval=20):
    model.train()
    total_psnr, total_count = 0, 0
    losses = []
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(labels, outputs)
        losses.append(loss.item())

        # backward
        loss.backward()
        optimizer.step()

        total_psnr += peak_signal_noise_ratio(outputs, labels)
        total_count += 1

        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:5.2f}".format(
                    epoch, idx, len(train_loader), total_psnr / total_count
                )
            )
            total_psnr, total_count = 0, 0
    epoch_psnr = total_psnr / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_psnr, epoch_loss


def train_epoch(model, optimizer, criterion, train_loader, device, epoch=0, log_interval=20):
    model.train()
    total_psnr, total_count = 0, 0
    losses = []
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(labels, outputs)
        losses.append(loss.item())

        # backward
        loss.backward()
        optimizer.step()

        total_psnr += peak_signal_noise_ratio(outputs, labels)
        total_count += 1

        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:5.2f}".format(
                    epoch, idx, len(train_loader), total_psnr / total_count
                )
            )
            total_psnr, total_count = 0, 0
    epoch_psnr = total_psnr / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_psnr, epoch_loss


def valid_epoch(model, criterion, val_loader, device):
    model.eval()
    total_psnr, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            total_psnr += peak_signal_noise_ratio(predictions, labels)
            total_count += 1

    epoch_psnr = total_psnr / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_psnr, epoch_loss


def training(model, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    train_psnrs, train_losses = [], []
    eval_psnrs, eval_losses = [], []

    for epoch in range(1, num_epochs + 1):
        time_start = time.time()
        train_psnr, train_loss = train_epoch(model, optimizer, criterion, train_loader, device, epoch)
        val_psnr, val_loss = valid_epoch(model, criterion, val_loader, device)
        train_psnrs.append(train_psnr.cpu())
        train_losses.append(train_loss)
        eval_psnrs.append(val_psnr.cpu())
        eval_losses.append(val_loss)

        if epoch % 10 == 0:
            inputs, labels = next(iter(val_loader))
            generate_images(model, inputs, labels)

        print("-" * 60)
        print(
            "| End of epoch: {:3d} | Time: {:5.2f}s | Train psnr: {:5.3f} | Train loss: {:5.3f} "
            "| Val psnr: {:5.3f} | Val loss: {:5.3f} ".format(
                epoch, time.time() - time_start, train_psnr, train_loss, val_psnr, val_loss
            )
        )
        print("-" * 60)
    model.eval()
    metrics = {
        "train_psnr": train_psnrs,
        "train_losses": train_losses,
        "val_psnr": eval_psnrs,
        "val_losses": eval_losses
    }
    return model, metrics
