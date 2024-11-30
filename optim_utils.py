import torch
from torchvision import transforms
from datasets import load_dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter, ImageEnhance
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
import scipy
import io


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)
    

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


# distortion_strength_paras = dict(
#     resizedcrop=(1, 0.5),
#     erasing=(0, 0.25),
#     contrast=(1, 2),
#     noise=(0, 0.1),
#     compression=(90, 10),
# )


# def relative_strength_to_absolute(strength, distortion_type):
#     assert 0 <= relative_strength <= 1
#     strength = (
#         strength
#         * (
#             distortion_strength_paras[distortion_type][1]
#             - distortion_strength_paras[distortion_type][0]
#         )
#         + distortion_strength_paras[distortion_type][0]
#     )
#     strength = max(strength, min(*distortion_strength_paras[distortion_type]))
#     strength = min(strength, max(*distortion_strength_paras[distortion_type]))
#     return strength


# def apply_distortion(
#     images,
#     distortion_type,
#     strength=None,
#     distortion_seed=0,
#     same_operation=False,
#     relative_strength=True,
#     return_image=True,
# ):
#     # Convert images to PIL images if they are tensors
#     if not isinstance(images[0], Image.Image):
#         images = to_pil(images)
#     # Check if strength is relative and convert if needed
#     if relative_strength:
#         strength = relative_strength_to_absolute(strength, distortion_type)
#     # Apply distortions
#     distorted_images = []
#     seed = distortion_seed
#     for image in images:
#         distorted_images.append(
#             apply_single_distortion(
#                 image, distortion_type, strength, distortion_seed=seed
#             )
#         )
#         # If not applying the same distortion, increment the seed
#         if not same_operation:
#             seed += 1
#     # Convert to tensors if needed
#     if not return_image:
#         distorted_images = to_tensor(distorted_images)
#     return distorted_images


# def apply_single_distortion(image, distortion_type, strength=None, distortion_seed=0):
#     # Accept a single image
#     assert isinstance(image, Image.Image)
#     # Set the random seed for the distortion if given
#     set_random_seed(distortion_seed)
#     # Assert distortion type is valid
#     assert distortion_type in distortion_strength_paras.keys()
#     # Assert strength is in the correct range
#     if strength is not None:
#         assert (
#             min(*distortion_strength_paras[distortion_type])
#             <= strength
#             <= max(*distortion_strength_paras[distortion_type])
#         )

#     # Apply the distortion
#     if distortion_type == "resizedcrop":
#         scale = (
#             strength
#             if strength is not None
#             else random.uniform(*distortion_strength_paras["resizedcrop"])
#         )
#         i, j, h, w = T.RandomResizedCrop.get_params(
#             image, scale=(scale, scale), ratio=(1, 1)
#         )
#         distorted_image = F.resized_crop(image, i, j, h, w, image.size)

#     elif distortion_type == "erasing":
#         scale = (
#             strength
#             if strength is not None
#             else random.uniform(*distortion_strength_paras["erasing"])
#         )
#         image = to_tensor([image], norm_type=None)
#         i, j, h, w, v = T.RandomErasing.get_params(
#             image, scale=(scale, scale), ratio=(1, 1), value=[0]
#         )
#         distorted_image = F.erase(image, i, j, h, w, v)
#         distorted_image = to_pil(distorted_image, norm_type=None)[0]

#     elif distortion_type == "contrast":
#         factor = (
#             strength
#             if strength is not None
#             else random.uniform(*distortion_strength_paras["contrast"])
#         )
#         enhancer = ImageEnhance.Contrast(image)
#         distorted_image = enhancer.enhance(factor)

#     elif distortion_type == "noise":
#         std = (
#             strength
#             if strength is not None
#             else random.uniform(*distortion_strength_paras["noise"])
#         )
#         image = to_tensor([image], norm_type=None)
#         noise = torch.randn(image.size()) * std
#         distorted_image = to_pil((image + noise).clamp(0, 1), norm_type=None)[0]

#     elif distortion_type == "compression":
#         quality = (
#             strength
#             if strength is not None
#             else random.uniform(*distortion_strength_paras["compression"])
#         )
#         quality = int(quality)
#         buffered = io.BytesIO()
#         image.save(buffered, format="JPEG", quality=quality)
#         distorted_image = Image.open(buffered)

#     else:
#         assert False

#     return distorted_image
def normalize_tensor(images, norm_type):
    assert norm_type in ["imagenet", "naive"]
    # Two possible normalization conventions
    if norm_type == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean, std)
    elif norm_type == "naive":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = transforms.Normalize(mean, std)
    else:
        assert False
    return torch.stack([normalize(image) for image in images])

def unnormalize_tensor(images, norm_type):
    assert norm_type in ["imagenet", "naive"]
    # Two possible normalization conventions
    if norm_type == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        unnormalize = transforms.Normalize(
            (-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
            (1 / std[0], 1 / std[1], 1 / std[2]),
        )
    elif norm_type == "naive":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unnormalize = transforms.Normalize(
            (-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
            (1 / std[0], 1 / std[1], 1 / std[2]),
        )
    else:
        assert False
    return torch.stack([unnormalize(image) for image in images])

def to_tensor(images, norm_type="naive"):
    assert isinstance(images, list) and all(
        [isinstance(image, Image.Image) for image in images]
    )
    images = torch.stack([transforms.ToTensor()(image) for image in images])
    if norm_type is not None:
        images = normalize_tensor(images, norm_type)
    return images

def to_pil(images, norm_type="naive"):
    assert isinstance(images, torch.Tensor)
    if norm_type is not None:
        images = unnormalize_tensor(images, norm_type).clamp(0, 1)
    return [transforms.ToPILImage()(image) for image in images.cpu()]

def image_distortion(img1, img2, seed, args):
    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)

    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img2)
        
    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)
    
    ### test more perturbation on tree ring
    if args.resizedcrop_factor_x is not None and args.resizedcrop_factor_y is not None:
        print("DBG", adsa) # test_debug
        scale_x = args.resizedcrop_factor_x if args.resizedcrop_factor_x is not None else random.uniform(1, 0.5)
        scale_y = args.resizedcrop_factor_y if args.resizedcrop_factor_y is not None else random.uniform(1, 0.5)
        i, j, h, w = T.RandomResizedCrop.get_params(
            img1, scale=(scale_x, scale_y), ratio=(1, 1)
        )
        img1 = F.resized_crop(img1, i, j, h, w, img1.size)
        img2 = F.resized_crop(img2, i, j, h, w, img2.size)

    if args.erasing_factor is not None:
        scale = args.erasing_factor if args.erasing_factor is not None else randtom.uniform(0, 0.25)
        img1 = to_tensor([img1], norm_type=None)
        img2 = to_tensor([img2], norm_type=None)
        i, j, h, w, v = T.RandomErasing.get_params(
            img1, scale=(scale, scale), ratio=(1, 1), value=[0]
        )
        img1 = F.erase(img1, i, j, h, w, v)
        img1 = to_pil(img1, norm_type=None)[0]
        img2 = F.erase(img2, i, j, h, w, v)
        img2 = to_pil(img2, norm_type=None)[0]

    if args.contrast_factor is not None:
        factor = args.contrast_factor if args.contrast_factor is not None else random.uniform(1, 2)
        enhancer = ImageEnhance.Contrast(img1)
        img1 = enhancer.enhance(factor)
        enhancer = ImageEnhance.Contrast(img2)
        img2 = enhancer.enhance(factor)


    if args.noise_factor is not None:
        std = args.noise_factor if args.noise_factor is not None else random.uniform(0, 0.1)
        img1 = to_tensor([img1], norm_type=None)
        noise = torch.randn(img1.size()) * std
        img1 = to_pil((img1 + noise).clamp(0, 1), norm_type=None)[0]
        img2 = to_tensor([img2], norm_type=None)
        img2 = to_pil((img2 + noise).clamp(0, 1), norm_type=None)[0]

    # if args.compression_factor is not None:
    #     quality = args.compression_factor if args.compression_factor is not None else random.uniform(90, 10)
    #     quality = int(quality)
    #     buffered = io.BytesIO()
    #     img1.save(buffered, format="JPEG", quality=quality)
    #     buffered.seek(0)
    #     img1 = img1.open(buffered)
    #     buffered = io.BytesIO()
    #     img2.save(buffered, format="JPEG", quality=quality)
    #     buffered.seek(0)
    #     img2 = img2.open(buffered)

    return img1, img2


# for one prompt to multiple images
def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return (image_features @ text_features.T).mean(-1)


def get_dataset(args):
    if 'laion' in args.dataset:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(args.dataset)['test']
        prompt_key = 'Prompt'

    return dataset, prompt_key


def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2

# def get_centered_tree_rings_mask_with_center(size, r):
#     """
#     Generates a mask with circular regions near the center and one at the center.
    
#     Parameters:
#     - size (int): Size of the square mask.
#     - r (int): Radius of the tree rings.
    
#     Returns:
#     - np.ndarray: Mask with tree rings near the center and at the center.
#     """
#     mask = np.zeros((size, size), dtype=bool)
    
#     # Adjust offsets to place the rings closer to the center
#     offset = size // 3  # Place rings at 1/3 and 2/3 of the size
#     centered_offsets = [
#         (offset, offset),            # Top-left (closer to center)
#         (offset, size - offset),     # Top-right (closer to center)
#         (size - offset, offset),     # Bottom-left (closer to center)
#         (size - offset, size - offset),  # Bottom-right (closer to center)
#         (size // 2, size // 2)       # Center
#     ]
    
#     for x_offset, y_offset in centered_offsets:
#         mask |= circle_mask(size, r=r, x_offset=x_offset - size // 2, y_offset=y_offset - size // 2)
    
#     return mask


def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)
    

    if args.w_mask_shape == 'circle':
        np_mask_center = circle_mask(init_latents_w.shape[-1], r=args.w_radius)
        # the first experiment on four rings: (failed with only showing the center one)
        np_mask_centerupright = circle_mask(init_latents_w.shape[-1], r=args.w_radius,x_offset = init_latents_w.shape[-1]//3,y_offset = init_latents_w.shape[-1]//3)
        np_mask_centerupleft = circle_mask(init_latents_w.shape[-1], r=args.w_radius,x_offset = -init_latents_w.shape[-1]//3,y_offset = init_latents_w.shape[-1]//3)
        np_mask_centerdownright = circle_mask(init_latents_w.shape[-1], r=args.w_radius,x_offset = init_latents_w.shape[-1]//3,y_offset = -init_latents_w.shape[-1]//3)
        np_mask_centerdownleft = circle_mask(init_latents_w.shape[-1], r=args.w_radius,x_offset = -10,y_offset = -10)
        np_mask= np_mask_centerdownleft # | np_mask_centerupright | np_mask_centerupleft | np_mask_centerdownright | np_mask_centerdownleft
        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    # if args.w_mask_shape == 'circle':
    #     np_mask = get_centered_tree_rings_mask_with_center(init_latents_w.shape[-1], r=args.w_radius)
    #     torch_mask = torch.tensor(np_mask).to(device)
    #     watermarking_mask[:, args.w_channel] = torch_mask
        
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask


def get_watermarking_pattern(pipe, args, device, shape=None):
    set_random_seed(args.w_seed)
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()

    if 'seed_ring' in args.w_pattern:
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -args.w_radius_incr):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    elif 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    for ma, gt in zip(watermarking_mask, gt_patch):
        
        if args.w_injection == 'complex':
            init_latents_w_fft[ma] = gt[ma].clone()
        elif args.w_injection == 'seed':
            init_latents_w[ma] = gt[ma].clone()
            return init_latents_w
        else:
            NotImplementedError(f'w_injection: {args.w_injection}')

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    return no_w_metric, w_metric

def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])
    
    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w
