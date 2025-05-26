import torch
from PIL import Image
import random
import logging
import torchvision

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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
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


def load_image(image_file, pil_image=None, input_size=224, max_num=12):
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
        targets = torch.stack([s['target'] for s in batch if s is not None])
        samples = torch.stack([s['samples'] for s in batch if s is not None])
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

            input_paths = random.choices([p[0] for p in self.samples if p != target_path and class_type in p], k=self.k)
            assert len(input_paths) == self.k # I think it may do this by default...
            samples = torch.stack([torch.from_numpy(self.processor(self.loader(i)).data['pixel_values'][0]) for i in input_paths])
        except Exception as e:
            logging.warning('getitem issue ', e)
            samples, target = None, None

        drop_mask = torch.rand(samples.shape[0],) < .2
        samples[drop_mask] = 0

        drop_whole_set_mask = torch.rand(1,) < .1
        if drop_whole_set_mask:
            samples = torch.zeros_like(samples)


        return {'samples': samples[:, :3], 'target': target[:3]}

    def __getitem__(self, index: int):
        return self.safe_getitem(index)


# https://data.mendeley.com/datasets/fs4k2zc5j5/3
# Gomez, J. C., Ibarra-Manzano, M. A., & Almanza-Ojeda, D. L. (2017). User Identification in Pinterest Through the Refinement of Cascade Fusion of Text and Images. Research in Computing Science, 144, 41-52.
def get_dataset(data_path, processor):
    return ImageFolderSample(data_path, 12, processor)


def get_dataloader(data_path, batch_size, num_workers, processor):
    dataloader = torch.utils.data.DataLoader(get_dataset(data_path, processor=processor), num_workers=num_workers, collate_fn=my_collate, batch_size=batch_size)
    return dataloader


