import segmentation_models_pytorch as smp


CLASS_NAMES = [
    "Road / Impervious",
    "Vegetation",
    "Building",
    "Unlabeled",
    "Land / Other",
]
CLASS_COLORS = [
    (14, 135, 204),
    (124, 252, 0),
    (155, 38, 182),
    (169, 169, 169),
    (255, 20, 147),
]
N_CLASSES = len(CLASS_NAMES)
ENCODER_NAME = "inceptionv4"
DECODER_CHANNELS = [256, 128, 64, 32, 16]


def build_model(
    n_classes: int = N_CLASSES,
    encoder_weights: str | None = "imagenet",
):
    """Build the transfer-learning Inception-v4 U-Net used by train and API."""
    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=encoder_weights,
        classes=n_classes,
        activation=None,
        encoder_depth=5,
        decoder_channels=DECODER_CHANNELS,
    )


def set_encoder_trainable(model, trainable: bool) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = trainable
