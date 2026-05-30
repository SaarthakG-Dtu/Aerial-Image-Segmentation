import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("outputs", "matplotlib"))

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .datagen import DataGen, TestDataGen
from .train import train
from .utils import plot_loss_vs_epoch, plot_iou_score_vs_epoch, plot_accuracy_vs_epoch, predict_image_mask_miou, DiceCELoss
from .model_factory import N_CLASSES, build_model, set_encoder_trainable

def main():
    parser = argparse.ArgumentParser(description="Aerial Image Segmentation Training Pipeline")
    parser.add_argument("--img_dir", type=str, default="classes_dataset/original_images/", help="Directory containing original images")
    parser.add_argument("--mask_dir", type=str, default="classes_dataset/label_images_semantic/", help="Directory containing mask images")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img_height", type=int, default=512, help="Image height")
    parser.add_argument("--img_width", type=int, default=768, help="Image width")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Directory to save outputs")
    parser.add_argument("--encoder_weights", type=str, default="imagenet", choices=["imagenet", "none"], help="Use ImageNet encoder weights for transfer learning")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0, help="Freeze the pretrained encoder for the first N epochs")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare data
    if not os.path.exists(args.img_dir):
        print(f"Error: Image directory {args.img_dir} not found.")
        return

    filenames = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not filenames:
        print("No images found in the dataset directory.")
        return

    # Split data similar to notebook
    X_train_and_val, X_test = train_test_split(filenames, test_size=0.1, random_state=19)
    X_train, X_val = train_test_split(X_train_and_val, test_size=0.15, random_state=19)

    print('Train size:', len(X_train))
    print('Validation size:', len(X_val))
    print('Test size:', len(X_test))

    # Transforms and datasets (using pure PyTorch v2)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = v2.Compose([
        v2.Resize((args.img_height, args.img_width), interpolation=v2.InterpolationMode.NEAREST), 
        v2.RandomHorizontalFlip(p=0.5), 
        v2.ElasticTransform(alpha=50.0, sigma=5.0) 
    ])
    val_transform = v2.Compose([
        v2.Resize((args.img_height, args.img_width), interpolation=v2.InterpolationMode.NEAREST), 
    ])

    train_dataset = DataGen(args.img_dir, args.mask_dir, X_train, mean, std, transform=train_transform, patch=False)
    val_dataset = DataGen(args.img_dir, args.mask_dir, X_val, mean, std, transform=val_transform, patch=False)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model definition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights
    model = build_model(n_classes=N_CLASSES, encoder_weights=encoder_weights)
    if args.freeze_encoder_epochs > 0:
        set_encoder_trainable(model, False)
    model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = DiceCELoss(weight_ce=1.0, weight_dice=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )

    # Training
    print("Starting training...")
    history = train(
        epochs=args.epochs,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=args.output_dir,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        patch=False
    )

    # Save final model state_dict
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Generate plots and save in output directory
    print("Generating performance plots...")
    plot_loss_vs_epoch(history, save_path=os.path.join(args.output_dir, "loss_vs_epoch.png"))
    plot_iou_score_vs_epoch(history, save_path=os.path.join(args.output_dir, "miou_vs_epoch.png"))
    plot_accuracy_vs_epoch(history, save_path=os.path.join(args.output_dir, "accuracy_vs_epoch.png"))
    
    # Inference on Test Set
    if args.epochs > 0 and len(X_test) > 0:
        print("Evaluating on test set sample...")
        t_test = v2.Resize((args.img_height, args.img_width), interpolation=v2.InterpolationMode.NEAREST)
        test_set = TestDataGen(args.img_dir, args.mask_dir, X_test, transform=t_test)
        
        sample_idx = min(3, len(test_set) - 1)
        image, mask = test_set[sample_idx]
        
        pred_mask, score = predict_image_mask_miou(model, image, mask, mean=mean, std=std)
        
        # Plot and save
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
        # Note: image might be CHW if it's returned as tensor, but test datagen returns PIL
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(mask)
        ax2.set_title('Ground Truth Mask')
        ax2.axis('off')
        
        ax3.imshow(pred_mask)
        ax3.set_title(f'Predicted Mask (mIoU: {score:.3f})')
        ax3.axis('off')

        inference_plot_path = os.path.join(args.output_dir, "inference_sample.png")
        plt.savefig(inference_plot_path, bbox_inches='tight')
        plt.close()
        print(f"Inference sample saved to {inference_plot_path}")

if __name__ == "__main__":
    main()
