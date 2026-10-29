import jax
import jax.numpy as jnp

import numpy as np

from torchvision import transforms

import matplotlib.pyplot as plt


def jax_has_gpu():
    print(jax.lib.xla_bridge.get_backend().platform)
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        print(True)
    except:
        print(False)


## Dataset Initialization
def transform_func():
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])


def transform_func_normalize():
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])


def transform_func_normalize_numpy():
    mean = np.array([0.485, 0.456, 0.406])[None, None, :]
    std = np.array([0.229, 0.224, 0.225])[None, None, :]
    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img = (img - mean) / std
        return img

    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), image_to_numpy])

def transform_func_numpy():
    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        return img

    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), image_to_numpy])

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

## Save images

def save_images(ori_img, adv_img, filename):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))  # Adjust figure size

    # Display the images
    axes[0].imshow(ori_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(adv_img)
    axes[1].set_title("Adversarial")
    axes[1].axis("off")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)