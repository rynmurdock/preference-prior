import torch
from PIL import Image
import random
import logging
import torchvision
import os
import glob

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=8, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, pil_image=None, input_size=224,):
    if not pil_image:
        pil_image = Image.open(image_file)
    image = pil_image.convert('RGB')
    transform = build_transform(input_size=input_size)
    # images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in [image]]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def my_collate(batch):
    try:
        targets = torch.stack([s['target'] for s in batch])
        samples = torch.stack([s['samples'] for s in batch])

        # targets = torch.stack([s['target'] for s in batch if s is not None])
        # samples = torch.stack([s['samples'] for s in batch if s is not None])
    except Exception as e:
      logging.warning('my_collate issue ', e)
      return None
    return samples, targets


class ImageFolderSample(torchvision.datasets.ImageFolder):
    def __init__(self, data_path, k, processor):
        super().__init__(data_path)
        self.k = k
        self.processor = processor

    def safe_getitem(self, index):
        try:
            target_path, class_type = self.samples[index]
            target = torch.from_numpy(self.processor(self.loader(target_path)).data['pixel_values'][0])

            # not sure why this is necessary
            if not isinstance(target, torch.Tensor):
                target = target[0]

            input_paths = random.choices([p[0] for p in self.samples if p != target_path and class_type in p], k=self.k)
            assert len(input_paths) == self.k # I think it may do this by default...
            samples = torch.stack([torch.from_numpy(self.processor(self.loader(i)).data['pixel_values'][0]) for i in input_paths])
            return {'samples': samples[:, :3], 'target': target[:3]}
        except Exception as e:
            logging.warning(f'getitem error: {e}')            
            return self.__getitem__(random.randint(0, len(self)-1))

    def __getitem__(self, index: int):
        return self.safe_getitem(index)


# https://data.mendeley.com/datasets/fs4k2zc5j5/3
# Gomez, J. C., Ibarra-Manzano, M. A., & Almanza-Ojeda, D. L. (2017). User Identification in Pinterest Through the Refinement of Cascade Fusion of Text and Images. Research in Computing Science, 144, 41-52.
def get_dataset(data_path, processor, k):
    return ImageFolderSample(data_path, k, processor,)


def is_dir_empty(path):
    with os.scandir(path) as scan:
        return next(scan, None) is None

# clean up any empty directories
def remove_empty_dirs(path_to_folders):
    folders = glob.glob(f'{path_to_folders}/*')
    for f in folders:
        if os.path.isdir(f) and is_dir_empty(f):
            os.rmdir(f)

def get_dataloader(data_path, batch_size, num_workers, processor, k):
    n_val_groups = 4
    val_batch_size = batch_size

    # we can die if we don't clean empty folders.
    remove_empty_dirs(data_path)
    
    full_data = get_dataset(data_path, processor=processor, k=k)

    # subset specific "classes" (subfolders that contain groups of preferred images)
    val_classes = random.sample(full_data.classes, k=n_val_groups)
    val_class_indices = []
    for cl in val_classes:
        val_class_indices.append(full_data.class_to_idx[cl])
    val_indices = [ind for ind, i in enumerate(full_data.samples) if i[1] in val_class_indices]

    train_indices = [i for i in range(len(full_data)) if i not in val_indices]
    train_data = torch.utils.data.Subset(full_data, train_indices)
    val_data = torch.utils.data.Subset(full_data, val_indices)


    train_dataloader = torch.utils.data.DataLoader(
                                            train_data, 
                                            num_workers=num_workers, 
                                            collate_fn=my_collate, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            drop_last=True
                                            )
    assert len(val_indices) >= val_batch_size
    val_dataloader = torch.utils.data.DataLoader(
                                            val_data, 
                                            num_workers=num_workers, 
                                            collate_fn=my_collate, 
                                            batch_size=val_batch_size, 
                                            drop_last=True
                                            )
    return train_dataloader, val_dataloader


