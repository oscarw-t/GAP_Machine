import dataclasses
import torchvision


@dataclasses.dataclass
class GeometricAugmentation:

    horiz_flip: float = 0.5
    vert_flip: float = 0.5
    rotation: int = 15


@dataclasses.dataclass
class ResolutionAugmentation:

    size: tuple[int, int] = (224, 224)


@dataclasses.dataclass
class PhotometricAugmentation:

    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05


class DatasetAugmentation:

    def __init__(
        self,
        geometric_aug: GeometricAugmentation,
        resolution_aug: ResolutionAugmentation,
        photometric_aug: PhotometricAugmentation,
    ):
        self.geometric_aug = geometric_aug
        self.resolution_aug = resolution_aug
        self.photometric_aug = photometric_aug

    def transform_train_dataset(self):

        return torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(
                    p=self.geometric_aug.horiz_flip
                ),
                torchvision.transforms.RandomVerticalFlip(
                    p=self.geometric_aug.vert_flip
                ),
                torchvision.transforms.RandomRotation(
                    degrees=self.geometric_aug.rotation
                ),
                torchvision.transforms.ColorJitter(
                    brightness=self.photometric_aug.brightness,
                    contrast=self.photometric_aug.contrast,
                    saturation=self.photometric_aug.saturation,
                    hue=self.photometric_aug.hue,
                ),
                torchvision.transforms.RandomResizedCrop(size=self.resolution_aug.size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def transform_val_test_dataset(self):

        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=self.resolution_aug.size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
