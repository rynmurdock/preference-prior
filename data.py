import torch
from PIL import Image
import random
import logging
import torchvision
import os
import glob
import json

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
        target_pixels = torch.stack([s['target_pixels'] for s in batch])
        sample_pixels = torch.stack([s['sample_pixels'] for s in batch])

        target_scores = torch.stack([torch.tensor(s['target_scores']) 
                                                  for s in batch])
        sample_scores = torch.stack([torch.tensor(s['sample_scores']) 
                                                  for s in batch])
        sample_prompts = [s['sample_prompts'] for s in batch]
        input_prompts = [s['input_prompt'] for s in batch]

        user_ages = torch.tensor([s['user_age'] for s in batch], dtype=torch.float32)
        user_genders = [s['user_gender'] for s in batch]
        user_nationalities = [s['user_nationality'] for s in batch]

    except Exception as e:
      logging.warning('my_collate issue ', e)
      return None
    return {
      "sample_pixels": sample_pixels,
      "sample_scores": sample_scores,
      "target_pixels": target_pixels,
      "target_scores": target_scores,
      "sample_prompts": sample_prompts,
      "input_prompts": input_prompts,
      "user_ages": user_ages,
      "user_genders": user_genders,
      "user_nationalities": user_nationalities,
  }

def process_json_to_dicts(json_data):
    pids_to_each_path_and_score = {}
    for s in json_data:
        # interaction information
        inter_info = [(s['image_path'], int(s['original_score']), s['image_metadata']['prompt'], s['user_demographics']['age'], 
                        s['user_demographics']['gender'], s['user_demographics']['nationality'])]
        # if we don't already have the participant's list of (image, score)s,
        #   let's add it
        if not pids_to_each_path_and_score.get(s['participant_id'], False):
            pids_to_each_path_and_score[s['participant_id']] = inter_info
        # otherwise, continue adding to their list
        else:
            pids_to_each_path_and_score[s['participant_id']] += inter_info
    return [v for v in pids_to_each_path_and_score.values()]

class ImageFolderSample(torch.utils.data.Dataset):
    def __init__(self, data_path, k, image_processor,):
        super().__init__()
        self.k = k
        self.image_processor = image_processor

        with open(data_path) as jsfile:
            self.json_data = json.load(jsfile)
        self.samples = process_json_to_dicts(self.json_data)
        self.data_path = os.path.dirname(data_path)

    def loader(self, img_p):
        im = Image.open(img_p)
        return im

    def __len__(self):
        return len(self.samples)

    def safe_getitem(self, index):
        try:
            user_history = self.samples[index]
            # sample k indices from user's ratings & one separate target

            # preferred_inds = [i for i in range(len(sample)) if sample[i][1] > 3]


            # TODO pad cases where len(sample) < k / prepackage interactions
            # would want to do it at point of embeddings
            # so would need to return list of tensors/similar, not single tensor
            if len(user_history) < 2:
                raise ValueError("sample has fewer than 2 items")
            
            pid_target = random.choice(range(len(user_history)))
            candidate_inds = [i for i in range(len(user_history)) if i != pid_target]
            
            if len(candidate_inds) >= self.k:
                pid_cond_subset = random.sample(candidate_inds, self.k)
            else:
                pid_cond_subset = random.choices(candidate_inds, k=self.k)

            target_path = user_history[pid_target][0]
            target_score = int(user_history[pid_target][1])
            # path set for PAMELA data relative to root
            target = self.image_processor(self.loader(
                f'{self.data_path}/../'+target_path)).data['pixel_values'][0]

            # not sure why this is necessary
            if not isinstance(target, torch.Tensor):
                target = target[0]

            assert len(pid_cond_subset) == self.k, f'{len(pid_cond_subset)=} != {self.k=}'
            
            # Demographics
            user_age = user_history[pid_target][3]
            user_gender = user_history[pid_target][4]
            user_nationality = user_history[pid_target][5]

            input_paths = [user_history[i][0] for i in pid_cond_subset]
            input_scores = [int(user_history[i][1]) for i in pid_cond_subset]
            input_prompt = user_history[pid_target][2]
            sample_prompts = [user_history[i][2] for i in pid_cond_subset]

            

            samples = torch.stack([self.image_processor(self.loader(f'{self.data_path}/../'+i)).data['pixel_values'][0] for i in input_paths])
            return {
                'sample_pixels': samples[:, :3],
                'target_pixels': target[:3],
                'sample_scores': input_scores,
                'target_scores': [target_score],
                'sample_prompts': sample_prompts,
                'input_prompt': input_prompt,
                'user_age': user_age,
                'user_gender': user_gender,
                'user_nationality': user_nationality
            }
        except Exception as e:
            logging.warning(f'getitem error: {e}')
            return self.__getitem__(random.randint(0, len(self)-1))

    def __getitem__(self, index: int):
        return self.safe_getitem(index)


# https://data.mendeley.com/datasets/fs4k2zc5j5/3
# Gomez, J. C., Ibarra-Manzano, M. A., & Almanza-Ojeda, D. L. (2017). User Identification in Pinterest Through the Refinement of Cascade Fusion of Text and Images. Research in Computing Science, 144, 41-52.
def get_dataset(data_path, image_processor, k):
    return ImageFolderSample(data_path, k, image_processor,)


def is_dir_empty(path):
    with os.scandir(path) as scan:
        return next(scan, None) is None

# clean up any empty directories
def remove_empty_dirs(path_to_folders):
    folders = glob.glob(f'{path_to_folders}/*')
    for f in folders:
        if os.path.isdir(f) and is_dir_empty(f):
            os.rmdir(f)

def get_dataloader(data_path, val_data_path, 
                   batch_size, num_workers, image_processor, k):
    val_batch_size = batch_size

    # we can die if we don't clean empty folders.
    remove_empty_dirs(data_path)
    
    train_data = get_dataset(data_path, image_processor=image_processor, k=k)
    val_data = get_dataset(val_data_path, image_processor=image_processor, k=k)

    # with pamela data, we have separate train/val sets
    # # subset specific "classes" (subfolders that contain groups of preferred images)
    # val_classes = random.sample(full_data.classes, k=n_val_groups)
    # val_class_indices = []
    # for cl in val_classes:
    #     val_class_indices.append(full_data.class_to_idx[cl])
    # val_indices = [ind for ind, i in enumerate(full_data.samples) if i[1] in val_class_indices]

    # train_indices = [i for i in range(len(full_data)) if i not in val_indices]
    # train_data = torch.utils.data.Subset(full_data, train_indices)
    # val_data = torch.utils.data.Subset(full_data, val_indices)

    train_dataloader = torch.utils.data.DataLoader(
                                            train_data, 
                                            num_workers=num_workers, 
                                            collate_fn=my_collate, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            drop_last=True
                                            )

    val_dataloader = torch.utils.data.DataLoader(
                                            val_data, 
                                            num_workers=num_workers, 
                                            collate_fn=my_collate, 
                                            batch_size=val_batch_size, 
                                            drop_last=True
                                            )
    return train_dataloader, val_dataloader
