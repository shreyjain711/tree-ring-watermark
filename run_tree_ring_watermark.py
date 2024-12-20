import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils_orig import *
from io_utils import *
from exp_utils import *


def main(args):
    table = None
    if args.with_tracking:
        wandb.login(key=args.wandb_API_key)
        if args.wandb_run_id is not None:
            wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'], id=args.wandb_run_id, resume='must')
        else:
            wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        # table = wandb.Table(columns=['gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'prompt', 'no_w_metric', 'w_metric', 'w_noise_vec', 'w_noise_vec_w_perturb', 'wm'])
        table = wandb.Table(columns=['exp_name', 'noise_vec_fft', 'noise_vec_fft_perturb', 'wm_mask', 'prompt', 'no_w_metric', 'w_metric', 'gen_no_w', 'gen_no_w_auged', 'gen_w', 'gen_w_auged'])
    

    exp_name = set_exp_name(args)

    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(pipe, args, device)

    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics = []

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        
        current_prompt = dataset[i][prompt_key] #if args.end!=1 else "white square background and a black square inside it center aligned and covering 80% of the area"
        
        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents()
        outputs_no_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
            )
        orig_image_no_w = outputs_no_w.images[0]
        
        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)


        # # Apply multiple watermarks

        # # Define positions for multiple watermarks
        # H, W = init_latents_w.shape[-2:]  # Height and width of the latent space
        # watermark_positions = [
        #     (H // 4, W // 4),        # Top-left
        #     (H // 4, 3 * W // 4),    # Top-right
        #     (3 * H // 4, W // 4),    # Bottom-left
        #     (3 * H // 4, 3 * W // 4) # Bottom-right
        # ]
    
        # for i in range(4):
        #     print(f"Applying watermark {i + 1}/4 at position {watermark_positions[i]}")
    
        #     # Generate a circular mask for this watermark
        #     x_offset, y_offset = watermark_positions[i]
        #     watermark_mask = circle_mask(
        #         size=max(H, W),
        #         r=args.w_radius,
        #         x_offset=x_offset - H // 2,
        #         y_offset=y_offset - W // 2
        #     )
        #     watermark_mask = torch.tensor(watermark_mask, dtype=torch.bool).to(device)
    
        #     # Generate the corresponding watermark pattern
        #     watermark_patch = get_watermarking_pattern(pipe, args, device)
    
        #     # Inject the watermark into the latent space
        #     init_latents_w = inject_watermark(init_latents_w, watermark_mask, watermark_patch, args)
    

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        # inject watermark
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
            )
        orig_image_w = outputs_w.images[0]

        ### test watermark
        # distortion
        orig_image_no_w_auged, orig_image_w_auged = image_distortion(orig_image_no_w, orig_image_w, seed, args)

        # reverse img without watermarking
        img_no_w = transform_img(orig_image_no_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # reverse img with watermarking
        img_w = transform_img(orig_image_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # eval
        no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)

        if args.reference_model is not None:
            sims = measure_similarity([orig_image_no_w, orig_image_w], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
            w_no_sim = sims[0].item()
            w_sim = sims[1].item()
        else:
            w_no_sim = 0
            w_sim = 0

        results.append({
            'no_w_metric': no_w_metric, 'w_metric': w_metric, 'w_no_sim': w_no_sim, 'w_sim': w_sim,
        })

        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)

        if args.with_tracking:
            #table: 'exp_name', 'noise_vec_fft', 'noise_vec_fft_perturb', 'wm_mask', 'prompt', 'no_w_metric', 'w_metric', 'gen_no_w', 'gen_no_w_auged', 'gen_w', 'gen_w_auged'
            noise_vec_fft_img, noise_vec_fft_perturb_img = wandb.Image(noise_vec_to_fft_real(init_latents_w)), wandb.Image(noise_vec_to_fft_real(reversed_latents_w))
            wm_mask_img = wandb.Image(watermarking_mask.to(torch.float32))
            #wm_mask_img = wandb.Image(gt_patch.to(torch.float32))

            gen_no_w_img, gen_no_w_auged_img = wandb.Image(orig_image_no_w), wandb.Image(orig_image_no_w_auged)
            gen_w_img, gen_w_auged_img = wandb.Image(orig_image_w), wandb.Image(orig_image_w_auged)

            table.add_data(exp_name, 
                           noise_vec_fft_img, noise_vec_fft_perturb_img, wm_mask_img, 
                           current_prompt, 
                           no_w_metric, w_metric, 
                           gen_no_w_img, gen_no_w_auged_img, 
                           gen_w_img, gen_w_auged_img)
            
            clip_scores.append(w_no_sim)
            clip_scores_w.append(w_sim)

    # roc
    preds = no_w_metrics +  w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'clip_score_mean': mean(clip_scores), 'clip_score_std': stdev(clip_scores),
                   'w_clip_score_mean': mean(clip_scores_w), 'w_clip_score_std': stdev(clip_scores_w),
                   'auc': auc, 'acc':acc, 'TPR@1%FPR': low})
    
    print(f'clip_score_mean: {mean(clip_scores)}')
    print(f'w_clip_score_mean: {mean(clip_scores_w)}')
    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int) # TODO test what happens on using -1, 0, 1, 2, 
    parser.add_argument('--w_pattern', default='ring')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    parser.add_argument('--w_radius_incr', default=1, type=int)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)
    parser.add_argument('--resizedcrop_factor_x', default=None, type=float)
    parser.add_argument('--resizedcrop_factor_y', default=None, type=float)
    parser.add_argument('--erasing_factor', default=None, type=float)
    parser.add_argument('--contrast_factor', default=None, type=float)
    parser.add_argument('--noise_factor', default=None, type=float)

    parser.add_argument('--wandb_API_key', default=None, type=str)
    parser.add_argument('--wandb_run_id', default=None, type=str)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)


### Mid sem testing reports
{
    'r_degree' : [15, 45, 60, 90, 135, 180],
    'jpeg_ratio' : [0.1, 0.2, 0.3, 0.5, 0.8],
    'crop_scale' : [0.1, 0.2, 0.3, 0.5, 0.8],
    'crop_ratio' : [0.2, 0.5, 1, 2, 5],
    'gaussian_blur_r' : [1, 2, 3, 5, 10],
    'gaussian_std' : [0.05, 0.1, 0.2, 0.3, 0.5],
    'brightness_factor' : [0.5, 0.8, 1.2, 1.5, 2.0],
    'contrast_factor' : [0.5, 0.8, 1.2, 1.5, 2.0],
    'resizedcrop_factor_x' : [0.2, 0.4, 0.6, 0.8, 1.0],
    'resizedcrop_factor_y' : [0.2, 0.4, 0.6, 0.8, 1.0],
    'erasing_factor' : [0.1, 0.2, 0.3, 0.4, 0.5],
    'noise_factor' : [0.1, 0.2, 0.3, 0.4, 0.5]
}

######### use default
# image_length
# model_id
# dataset
# guidance_scale
# w_seed
# w_channel
# w_pattern
# w_mask_shape
# test_num_inference_steps
# reference_model
# reference_model_pretrain
# max_num_log_image
# gen_seed
# w_measurement
# w_injection
# w_pattern_const
