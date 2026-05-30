import time
import os
from tqdm import tqdm
import numpy as np

import torch

from .model_factory import set_encoder_trainable
from .utils import mean_iou, pixel_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train(
    epochs,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    output_dir="outputs",
    freeze_encoder_epochs=0,
    patch=False,
):
    torch.cuda.empty_cache()
    losses_train = []
    losses_test = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    os.makedirs(output_dir, exist_ok=True)
    num_of_times_loss_not_improving = 0

    model.to(device)
    fit_time = time.time()
    for epoch in range(epochs):
        if freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs:
            set_encoder_trainable(model, True)
            print("Encoder unfrozen for fine-tuning.")

        start_time = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training Loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            image_tiles, mask_tiles = data
            if patch:
                batch_size, n_tiles, channel, height, width = image_tiles.size()
                image_tiles = image_tiles.view(-1, channel, height, width)
                mask_tiles = mask_tiles.view(-1, channel, height, width)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)

            # Forward Propagation
            predicted_image = model(image)
            loss = criterion(predicted_image, mask)

            # Metric to do Evaluation
            iou_score += mean_iou(predicted_image, mask)
            accuracy += pixel_accuracy(predicted_image, mask)

            # Backward Propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0

            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    image_tiles, mask_tiles = data

                    if patch:
                        batch_size, n_tiles, channel, height, width = image_tiles.size()
                        image_tiles = image_tiles.view(-1, channel, height, width)
                        mask_tiles = mask_tiles.view(-1, height, width)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)

                    # Forward Propagation
                    predicted_image = model(image)

                    # Metric to do Evaluation
                    val_iou_score += mean_iou(predicted_image, mask)
                    test_accuracy += pixel_accuracy(predicted_image, mask)

                    loss = criterion(predicted_image, mask)
                    test_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            val_loss = test_loss / len(val_loader)
            train_epoch_iou = iou_score / len(train_loader)
            train_epoch_acc = accuracy / len(train_loader)
            val_epoch_iou = val_iou_score / len(val_loader)
            val_epoch_acc = test_accuracy / len(val_loader)

            losses_train.append(train_loss)
            losses_test.append(val_loss)
            train_iou.append(train_epoch_iou)
            train_acc.append(train_epoch_acc)
            val_iou.append(val_epoch_iou)
            val_acc.append(val_epoch_acc)

            if min_loss > val_loss:
                print(
                    "Loss Decreasing... {:.3f} >> {:.3f} ".format(
                        min_loss, val_loss
                    )
                )
                min_loss = val_loss
                num_of_times_loss_not_improving = 0
                checkpoint_path = os.path.join(output_dir, "best_model.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved best checkpoint to {checkpoint_path}")
            else:
                num_of_times_loss_not_improving += 1
                print(f"Loss not improving for {num_of_times_loss_not_improving} epoch(s)")
                if num_of_times_loss_not_improving == 6:
                    print("Loss not improving for 6 epochs, stopping training.")
                    break

            print(
                "Epoch:{}/{}..".format(epoch + 1, epochs),
                "Train Loss:{:.3f}..".format(train_loss),
                "Validation Loss: {:.3f}..".format(val_loss),
                "Train mean_iou:{:.3f}..".format(train_epoch_iou),
                "Validation mean_iou: {:.3f}..".format(val_epoch_iou),
                "Train Acc:{:.3f}..".format(train_epoch_acc),
                "Val Acc:{:.3f}..".format(val_epoch_acc),
                "Time: {:.2f}m".format((time.time() - start_time) / 60),
            )

    history = {
        "train_loss": losses_train,
        "val_loss": losses_test,
        "train_miou": train_iou,
        "val_miou": val_iou,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "lrs": lrs,
    }
    print("Total time: {:.2f} m".format((time.time() - fit_time) / 60))
    return history
