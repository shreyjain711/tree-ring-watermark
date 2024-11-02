import torch

def set_exp_name(args):
    name_strs = [args.run_name]
    if args.num_inference_steps != 50:
        name_strs.append(f"NIS{args.num_inference_steps}")
    if args.w_pattern != 'ring':
        name_strs.append(f"P{args.w_pattern}")
    if args.w_radius != 10:
        name_strs.append(f"R{args.w_radius}")
    if args.w_radius_incr != 1:
        name_strs.append(f"R+={args.w_radius_incr}")

    perturbation_abbr_map = {
        'r_degree': 'RF', 
        'crop_scale': 'CS', 
        'crop_ratio': 'CR', 
        'gaussian_std': 'GsF', 
        'brightness_factor': 'BF', 
        'resizedcrop_factor_x': 'RxF', 
        'resizedcrop_factor_y': 'RyF', 
        'erasing_factor': 'EF', 
        'contrast_factor': 'CnF', 
        'noise_factor': 'NF', 
        'jpeg_ratio': 'CmF',
        'gaussian_blur_r': 'GbF'
    }
    
    for perturb in ['r_degree', 'crop_scale', 'crop_ratio', 'gaussian_std', 'brightness_factor', 'resizedcrop_factor', 'erasing_factor', 'contrast_factor', 'noise_factor']:
        if eval(f'args.{perturb}') is not None:
            name_strs.append(f"{perturbation_abbr_map[perturb]}{eval(f'args.{perturb}'):.3f}")

    for perturb in ['jpeg_ratio', 'gaussian_blur_r']:
        if eval(f'args.{perturb}') is not None:
            name_strs.append(f"{perturbation_abbr_map[perturb]}{eval(f'args.{perturb}'):d}")

    exp_name = "_".join(name_strs)
    return exp_name
    
def noise_vec_to_fft_real(noise_vec):
    return torch.fft.fftshift(torch.fft.fft2(noise_vec), dim=(-1, -2)).real.to(torch.float32)
