import os
import sys
import yaml
import argparse
from argparse import Namespace
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#load experiment config
config_file = 'config.yaml'
with open(config_file, 'r') as file:
    general_config = yaml.safe_load(file)
    
#command line arguments - update config.yaml 
if general_config['general']['config_source'] == 'cmd':
    print(" *** Applying configuration from commandline experiment parameters ***")
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cuda_device_id', default=1, type=int, choices=[0,1])
    # parser.add_argument('--seg_loss_name', default='GWD', type=str, choices=['DiceCELoss', 'DiceFocalLoss', 'GDL', 'GWD'])
    # parser.add_argument('--is_weighted_cls', default=True, action='store_true')
    # parser.add_argument('--weighting_mode', default='default', type=str, choices=['default', 'GDL'])
    # parser.add_argument('--classes', default=33, type=int, choices=[9, 17, 33])
    
    args_dict = vars(parser.parse_args())
    print(args_dict)
    print("----------------------------------")
    general_config['args'].update(args_dict)
 
args = argparse.Namespace(**general_config['args'])

if args.comet:
    import comet_ml
    from comet_ml import Experiment
else:
    from src.logger import DummyExperiment
    experiment = DummyExperiment()
    print("Comet logger false.")
    
import uuid
import builtins
import itertools
import json
import re
import glob
import random
import time
import warnings
import numpy as np
from tqdm import tqdm
from functools import partial
from datetime import datetime
from contextlib import contextmanager
from torch.utils.data._utils.collate import default_collate
warnings.filterwarnings("ignore", category=FutureWarning, module="monai.data.dataset")

# TORCH modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler
from torch.nn import MSELoss

#MONAI modules
from monai.losses import DiceLoss
from monai.metrics import MSEMetric
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric, DiceMetric
from monai.metrics import CumulativeAverage
# from monai.optimizers import WarmupCosineSchedule
from monai.data import ThreadDataLoader, DataLoader, decollate_batch
from monai.data.dataset import PersistentDataset, Dataset
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from PIL import Image
from monai.data.utils import collate_meta_tensor 

from src.cuda_setup import configure_cuda
from src.data_preparation import split_train_val, create_domain_labels, build_sampler, load_dataset_json
from src.model import DWNet
from src.scheduler import CosineAnnealingWarmupRestarts, WarmupCosineSchedule
from src.losses import DiceCELoss, DiceFocalLoss, AngularLoss, FocalDiceBCELoss, DiceBCELoss
from src.transforms import Transforms
from src.logger import DummyExperiment
from src.deep_watershed import deep_watershed_with_voting
from src.inference_utils import save_float_map, save_inference_multiclass_segmentation
from src.metrics_helper import MetricsHelper    
from src.log_helper import LogHelper

def namespace_to_dict(ns):
    if isinstance(ns, Namespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    else:
        return ns
    

def gpu_transform(train_data, trans, use_thread_loader=False):
    """
    GPU processing - applies GPU heavy transforms per sample.

    Parameters
    ----------
    train_data : tuple or dict
        Either a tuple where the first element contains the batch data, 
        or a dictionary with tensor batches.
    trans : object
        Transformation object with a `gpu_transform` method.
    use_thread_loader : bool, optional
        If True, bypasses this function and returns the original `train_data`.

    Returns
    -------
    torch.Tensor or dict
        Collated batch data after GPU transforms have been applied.
    """
    if use_thread_loader:
        return train_data

    batch_list = []
    # If tuple, unpack first element
    if isinstance(train_data, list):
        train_data = train_data[0]

    batch_size = train_data["image"].shape[0]

    for i in range(batch_size):
        sample = {k: v[i] for k, v in train_data.items()}
        sample = trans.gpu_transform(sample)  # apply GPU transforms per sample
        batch_list.append(sample)

    return default_collate(batch_list)


def collate_meta_tensor_with_crop(batch, transforms):
    processed_batch = []
    for sample in batch:
        sample = transforms(sample)
        processed_batch.append(sample)
    return collate_meta_tensor(processed_batch)


def merge_pulp_into_teeth(multiclass_pred: np.ndarray, pulp_pred: np.ndarray, pulp_class: int = 111, excluded_classes=None) -> np.ndarray:
    """
    Merge pulp segmentation into teeth predictions.
    
    Args:
        teeth_pred: 3D array of teeth instance predictions (e.g., from deep watershed)
        pulp_pred: 3D array of binary pulp segmentation (1 = pulp, 0 = background)
        pulp_class: integer label to assign to pulp voxels inside teeth (default=111)
        excluded_classes: list of class labels to exclude from teeth mask
    
    Returns:
        merged_pred: 3D array with pulp merged into teeth, respecting tooth mask
    """
    if excluded_classes is None:
        excluded_classes = list(range(0, 11)) + [43, 44, 45]
        
    merged_pred = multiclass_pred.copy()
    
    # Mask: valid teeth voxels (not in excluded classes)
    teeth_mask = ~np.isin(multiclass_pred, excluded_classes)
    
    # Only keep pulp inside valid teeth
    pulp_inside_teeth = (pulp_pred > 0) & teeth_mask
    
    # Assign pulp class
    merged_pred[pulp_inside_teeth] = pulp_class
    
    return merged_pred

def setup_training(args):
    
    #REPRODUCIBLITY
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # precision for nn.modules : eg. nn.conv3d - # Nvidia Ampere 
    # precision for linear algebra - eg. interpolations and elastic transforms
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if args.seed != -1:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        set_determinism(seed=args.seed)
        torch.use_deterministic_algorithms(mode=args.deterministic_algorithms, warn_only=True)
        if args.deterministic_algorithms:
            warnings.warn(
                "You have chosen to seed training. "
                "This will turn on the CUDNN deterministic setting, "
                "which can slow down your training considerably! "
                "You may see unexpected behavior when restarting "
                "from checkpoints."
            )

    # Average mixed precision settings, default to torch.float32
    scaler = None
    TORCH_DTYPES = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }
    autocast_d_type = TORCH_DTYPES[args.autocast_dtype]
    val_autocast_d_type = TORCH_DTYPES[args.inference_autocast_dtype]
    if args.use_scaler:
        scaler = torch.amp.GradScaler()
        if autocast_d_type == torch.bfloat16:
            # detect gradient errors - debug cuda C code
            os.environ["TORCH_CUDNN_V8_API_ENABLED"]="1"
        if autocast_d_type != torch.float32:
            torch.autograd.set_detect_anomaly(True)

    #CUDA
    configure_cuda(args.gpu_frac, num_threads=args.num_threads, device=args.device, visible_devices=args.visible_devices, use_cuda_with_id=args.cuda_device_id)
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device_id}")
    else:
        device = torch.device("cpu")

    #DATA CACHE
    args.cache_path = os.path.join(args.cache_dir, f"p_{args.patch_size[0]}_{args.patch_size[1]}_{args.patch_size[2]}")
    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)

    for subdir in ['train', 'val', 'test']:
        subdir_path = os.path.join(args.cache_path, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
                        
    if args.clear_cache:
        print("Clearing cache...")
        train_cache = glob.glob(os.path.join(args.cache_path, 'train/*.pt'))
        val_cache = glob.glob(os.path.join(args.cache_path, 'val/*.pt'))
        test_cache = glob.glob(os.path.join(args.cache_path, 'test/*.pt'))
        if len(train_cache) != 0:
            for file in train_cache:
                os.remove(file)
        if len(val_cache) != 0:
            for file in val_cache:
                os.remove(file)
        if len(test_cache) != 0:
            for file in test_cache:
                os.remove(file)
        print(f"Cleared cache in dir: {args.cache_path}, train: {len(train_cache)} files, val: {len(val_cache)} files, test: {len(test_cache)} files.")

    return scaler, autocast_d_type, val_autocast_d_type, device

def training_step(args, losses, batch_idx, epoch, model, optimizer, scaler, data_sample, train_loader,
                  autocast_d_type, device, pbar, metrics_helper, log_helper):
    with torch.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type, device_type=device.type):
        output = model(data_sample["image"])
        # unpack output
        (seg_multiclass, dist, direction, pulp) = output
        
        #loss
        multiclass_dice_loss, ce_loss = losses['multiclass_seg_loss'](seg_multiclass, data_sample["label"][:,0:1]) #primary labels - no pulp loss
        total_multiclass_loss = multiclass_dice_loss + ce_loss
        dist_loss = losses['dist_loss'](dist, data_sample["watershed_map"][:,0:1])
        dir_loss = losses['dir_loss'](direction, data_sample["watershed_map"][:,1:], torch.where(data_sample["label"][:,0:1].long() >= 1, 1, 0))
        pulp_loss = losses['pulp_loss'](pulp, data_sample["label"][:,1:])  #pulp labels
        loss =  total_multiclass_loss * args.loss_weights['multiclass_seg_loss'] + \
                dist_loss * args.loss_weights['dist_loss'] + \
                dir_loss *  args.loss_weights['dir_loss'] + \
                pulp_loss * args.loss_weights['pulp_loss']

        if args.use_scaler and scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # update metrics
        outputs = (seg_multiclass, pulp, dist)
        labels = (data_sample["label"][:,0:1], data_sample["label"][:,1:], data_sample["watershed_map"][:,0:1])
        metrics_helper.update(outputs, labels, epoch)

        loss_values = [total_multiclass_loss, dist_loss, dir_loss, pulp_loss]
        loss_dict = dict(zip(args.loss_names, loss_values))
        # update losses
        log_helper.update_losses(loss_dict)
        pbar.set_postfix({"loss": f"{log_helper.get_last_loss():.4f}"})


# INFERENCE STEP
def inference_step(args, batch_idx, epoch, model, scaler, data_sample, data_loader, autocast_d_type, device, pbar, trans, metric_helper, log_helper):
    with warnings.catch_warnings(), torch.amp.autocast(enabled=True, dtype=autocast_d_type, device_type=device.type):
        output = sliding_window_inference(data_sample["image"], roi_size=args.patch_size, sw_batch_size=4, predictor=model, 
                                          overlap=0.5, sw_device=device, device='cpu', mode='gaussian', sigma_scale=0.125,
                                          padding_mode='constant', cval=0, progress=False)
    
        #metrics
        (seg_multiclass, dist, direction, pulp) = output
        outputs = (seg_multiclass, pulp, dist)
        labels = (data_sample["label"][:,0:1], data_sample["label"][:,1:], data_sample["watershed_map"][:,0:1])
        # metric_helper.update(outputs, labels, epoch)

        #predictions
        # multiclass_segmentation = seg_multiclass.argmax(dim=1, keepdim=True).squeeze().cpu().numpy()
        # pulp_segmentation = (torch.sigmoid(pulp) > 0.5).float().squeeze().cpu().numpy()
        # dist_pred = nn.Threshold(1e-3, 0)(torch.sigmoid(dist)).squeeze().cpu().numpy()
        
        # pred_multiclass = deep_watershed_with_voting(dist_pred, multiclass_segmentation)
        # pred_final = merge_pulp_into_teeth(pred_multiclass, pulp_segmentation, pulp_class=111)
        # oracle
        # pred_multiclass = deep_watershed_with_voting(data_sample["watershed_map"][0, 0:1].squeeze().cpu().numpy(), data_sample["label"][0, 0:1].squeeze().cpu().numpy())
        # pred_final = merge_pulp_into_teeth(pred_multiclass, data_sample["label"][0, 1:].squeeze().cpu().numpy(), pulp_class=111)
        
        # data_sample["pred"] = dist.data
        # d = [trans.post_transform_binary(i) for i in decollate_batch(data_sample)] # IT WORKED !!!!
        
        # meta_dict = data_sample['image'].meta
        # scan_name = f"{epoch}_{meta_dict['filename_or_obj'][0].split('/')[-1].replace('.nii.gz', '')}"
        
        # sample = decollate_batch(data_sample)
        # dict_pred = {"image":sample[0]["image"], "pred":torch.from_numpy(dist)}
        # transformed_dist = trans.invert_inference_transform(dict_pred)["pred"]
    
        
        # save_float_map(output_dir="output/direction_map", array=dist, meta_dict=meta_dict,
        #                invert_transform=trans.invert_inference_transform, original_image=data_sample['image'],
        #                name='dist', dtype=np.float32)
        # save_float_map(output_dir="output/direction_map", array=direction, meta_dict=meta_dict,
        #                invert_transform=trans.invert_inference_transform, original_image=data_sample['image'],
        #                name='dir', dtype=np.float32)
        # save_float_map(output_dir="output/direction_map", array=pulp_segmentation, meta_dict=meta_dict,
        #                invert_transform=trans.invert_inference_transform, original_image=data_sample['image'],
        #                name='pulp', dtype=np.uint8)
        
        # save_inference_multiclass_segmentation(output_dir="output/multiclass_seg", array= multiclass_segmentation, meta_dict=meta_dict,
        #                                        invert_transform=trans.invert_inference_transform, original_image=data_sample['image'],
        #                                        is_invert_mapping=False, dtype=np.uint8)
        # save_inference_multiclass_segmentation(output_dir="output/multiclass_seg", array= pred_final, meta_dict=meta_dict,
        #                                        invert_transform=trans.invert_inference_transform, original_image=data_sample['image'],
        #                                        is_invert_mapping=True, dtype=np.uint8)
    
    
_builtin_print = builtins.print  # store the original print

@contextmanager
def use_tqdm_print(args=None):
    """
    Context manager to replace print with tqdm.write if args.use_tqdm_print is True.
    
    Args:
        args: Namespace or object with attribute `use_tqdm_print`. 
              If None, defaults to always using normal print.
    """
    if args is not None and getattr(args, "use_tqdm_print", False):
        def tqdm_print(*print_args, **kwargs):
            tqdm.write(" ".join(str(a) for a in print_args))

        builtins.print = tqdm_print
    try:
        yield
    finally:
        builtins.print = _builtin_print

def main():
    #load experiment configurations, setup cuda
    scaler, autocast_d_type, val_autocast_d_type, device = setup_training(args)
    
    #setup experiment and logger
    if args.comet:
        experiment = Experiment(project_name="tf3")
        unique_experiment_name = experiment.get_name()
        experiment_key = experiment.get_key()
    else:
        experiment = DummyExperiment()
        unique_experiment_name = uuid.uuid4().hex

        args.validation_interval = 5
        
        args.multiclass_metrics_firstTime_log = 10
        args.multiclass_metrics_interval = 5
        args.multiclass_metrics_epoch = 5
        args.binary_metrics_interval = 5
        args.distance_metrics_interval = 5
        
        args.log_slice_2d_interval = 1
        args.log_3d_scene_interval_training = 5
        args.log_3d_scene_interval_validation = 5
        
        
    #TODO add logger    
    logger = None
            
    print("--------------------")
    print (f"\n *** Starting experiment: {unique_experiment_name}:\n")
    print(f"Current server time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    #PREPARE DATASETS   
    dataset_json_path = os.path.join(args.data, "dataset.json")
    class_labels = load_dataset_json(dataset_json_path)

    train_data, val_data, all_data_paths = split_train_val(args)
    train_labels = create_domain_labels(train_data)
    val_labels = create_domain_labels(val_data)
    
    if args.generate_watershed_maps:
        from src.data_preparation import generate_watershed_maps
        args.watershed_maps_dir = os.path.join(args.data, args.watershed_maps_folder)
        watershed_files_paths = [os.path.join(args.watershed_maps_source_labels, f) for f in os.listdir(args.watershed_maps_source_labels) if f.endswith("_primary.nii.gz")]
        watershed_files_paths.sort()
        generate_watershed_maps(watershed_files_paths, args.watershed_maps_dir, device="cuda")

    trans=Transforms(args, device=device)
    
    if args.use_persistent_dataset:
        pre_cache_train_dataset = PersistentDataset(data=train_data, transform=trans.preprocessing_transforms, cache_dir=os.path.join(args.cache_path, 'train'))
        train_transform = trans.train_transform if args.use_thread_loader else trans.preprocessing_transforms
        train_dataset = PersistentDataset(data=train_data, transform=train_transform, cache_dir=os.path.join(args.cache_path, 'train'))
        val_dataset = PersistentDataset(data=val_data, transform=trans.inference_transform, cache_dir=os.path.join(args.cache_path, 'val'))
    else:
        # Your normal Dataset class expects list of filenames or similar
        train_dataset = Dataset(train_data, transform=trans.train_transform, root_dir=args.data)
        val_dataset = Dataset(val_data, transform=trans.inference_transform, root_dir=args.data)

    train_sampler = build_sampler(train_dataset, train_labels, args)
    val_sampler = build_sampler(val_dataset, val_labels, args)
    
    pre_cache_loader = DataLoader(pre_cache_train_dataset, num_workers=args.num_workers_cache, batch_size=1, sampler=train_sampler, pin_memory=False)
    
    if args.use_thread_loader:
        train_loader = ThreadDataLoader(train_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = ThreadDataLoader(val_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=val_sampler)
    else:
        train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, sampler=train_sampler, persistent_workers=True,
                                  pin_memory=args.pin_memory, worker_init_fn=np.random.seed(args.seed), prefetch_factor=args.prefetch_factor,
                                  collate_fn=partial(collate_meta_tensor_with_crop, transforms=trans.pre_collate_transform)) #collate after random crop - to enable batches
        val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size_val, sampler=val_sampler, pin_memory=args.pin_memory)
    
    #MODEL
    #num classes = 10 anatomical classes + 32 tooth classes + 3 canal classes + 1 pulp
    model = DWNet(spatial_dims=3, in_channels=1, out_channels=args.out_channels, act=args.activation, norm=args.norm,
                  bias=False, backbone_name=args.backbone_name, configuration=args.configuration)
    if args.parallel:
        model = nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.to(device)
    
    #LOSSES
    weights = None
    if args.weighting_mode != 'none':
        #TODO implement weights calculation, maybe json file?
        if args.weighting_mode == 'inverse_frequency_class_weights':
            weights = torch.tensor(class_labels['weights'], dtype=torch.float32, device=device)
        else:
            raise NotImplementedError(f"Weighting mode {args.weighting_mode} not implemented.")
        
    if args.seg_loss_name=="DiceCELoss":
        criterion_seg = DiceCELoss(include_background=args.include_background_loss, ce_weight=weights, to_onehot_y=True, softmax=True)
    elif args.seg_loss_name=="FocalDice":
        criterion_seg = DiceFocalLoss(include_background=args.include_background_loss, focal_weight=weights, to_onehot_y=True, softmax=True, gamma=args.wasserstein_distance_matrix)

    criterion_distance = MSELoss()
    criterion_direction = AngularLoss()
    # criterion_pulp = FocalDiceBCELoss(alpha=args.focal_alpha, gamma=args.focal_gamma, bce_weight=args.bce_weight)
    criterion_pulp = DiceBCELoss(alpha=args.focal_alpha, bce_weight=args.bce_weight)
        
    criteria = [criterion_seg, criterion_distance, criterion_direction, criterion_pulp]
    losses = dict(zip(args.loss_names, criteria)) #keep order of loss names!
    
    #OPTIMIZER
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.adam_ams, eps=args.adam_eps)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps)
    else:
        raise NotImplementedError(f"There are no implementation of: {args.optimizer}")

    # SCHEDULER
    if args.scheduler_name == 'cosine_annealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, verbose=True)
    elif args.scheduler_name == 'warmup_cosine':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs, min_lr=args.min_lr)
    elif args.scheduler_name == "warmup_cosine_restarts":
        scheduler = CosineAnnealingWarmupRestarts(optimizer, warmup_steps=args.warmup_steps, first_cycle_steps=int(
            args.epochs * args.first_cycle_steps), cycle_mult=0.5, gamma=args.scheduler_gamma, max_lr=args.lr, min_lr=args.min_lr)

    #LOAD PRETRAINED MODEL
    if args.continue_training:
        checkpoint_data = torch.load(args.trained_model, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        scheduler_state = checkpoint_data.get('scheduler_state_dict', None)
        if scheduler_state is None:
            scheduler_state = checkpoint_data.get('lr_scheduler', None)   
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        args.start_epoch = checkpoint_data['epoch']
        print(f'Loaded model, optimizer and scheduler - continue training from epoch: {args.start_epoch}')

    #METRICS
    metrics_helper = MetricsHelper(args)
    log_helper = LogHelper(experiment, args)

    #Setup print
    disable_tqdm = not sys.stderr.isatty()
    
    if args.create_preproc_cache:
        start_time_epoch = time.time()
        for train_data in tqdm(pre_cache_loader, desc=f"Cache persistent dataset", total=len(pre_cache_loader),
                               file=sys.stderr, position=1, leave=False, disable=disable_tqdm):
            continue
        cache_time=time.time() - start_time_epoch
        print(f"Pre-cache took: {cache_time:.2f}s.")
        
    for epoch in tqdm(range(args.start_epoch, args.epochs+1), desc="Epochs", file=sys.stderr,
                      position=0, leave=True, disable=disable_tqdm):
        
        ######################################################################################################
        ## TRAINING STEP
        model.train()
        start_time_epoch = time.time()
        pbar = tqdm(enumerate(train_loader), desc=f"Training (epoch {epoch})", total=len(train_loader),
                    file=sys.stderr, position=1, leave=False, disable=disable_tqdm)
        with use_tqdm_print(args.use_tqdm_print):
            for batch_idx, train_data in pbar:
                train_data = gpu_transform(train_data, trans, args.use_thread_loader)
                training_step(args, losses, batch_idx, epoch, model, optimizer, scaler, train_data, train_loader, 
                                autocast_d_type, device, pbar, metrics_helper, log_helper)
            metrics_helper.compute(epoch, phase="train")
            metrics_helper.reset()
            log_helper.log_metrics(epoch, metrics_helper.results, phase="train")
            log_helper.log_losses(epoch, phase="train")
            scheduler.step()
            epoch_time=time.time() - start_time_epoch
            print(f" Train loop finished - total time: {epoch_time:.2f}s.")
        ######################################################################################################
        
        
        ######################################################################################################
        ## VALIDATION (INFERENCE) STEP

        if (epoch) % args.validation_interval == 0 and epoch != 0 and args.run_validation:
            start_time_validation = time.time()
            pbar = tqdm(enumerate(val_loader), desc=f"Validation (epoch {epoch})", total=len(val_loader),
                    file=sys.stderr, position=1, leave=False, disable=disable_tqdm)
            model.eval()
            with use_tqdm_print(args.use_tqdm_print):
                for batch_idx, val_data in pbar:
                    with torch.no_grad():
                        inference_step(args, batch_idx, epoch, model, scaler, val_data, val_loader,
                                       val_autocast_d_type, device, pbar, trans, metrics_helper, log_helper)
                # metrics_helper.compute(epoch, phase="val")
                # metrics_helper.reset()
                # log_helper.log_metrics(epoch, metrics_helper.results, phase="val")
                val_time=time.time() - start_time_validation
                print( f"Validation time: {val_time:.2f}s")
                            
        # CHECKPOINTS SAVE
        if args.save_checkpoints:
            #create unique experiment name
            directory = f"checkpoints/{args.checkpoint_dir}/{unique_experiment_name}/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # save config to YAML
            config_path = os.path.join(directory, "config.yaml")
            with open(config_path, "w") as f:
                args_dict = namespace_to_dict(args)
                yaml.safe_dump(args_dict, f)
            
            
            # save model checkpoint
            if (epoch) % args.save_interval == 0 and epoch != 0:
                ckpt_path = os.path.join(directory, f"model_epoch_{epoch}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "current_train_metrics": metrics_helper.get_current("train"),
                    "current_val_metrics": metrics_helper.get_current("val"),
                    "best_train_metrics": metrics_helper.get_best("train"),
                    "best_val_metrics": metrics_helper.get_best("val"),
                    'experiment_name': unique_experiment_name,
                }, ckpt_path)
                print(f"Checkpoint saved.")
                    
            # save best TRAIN model
            # if (epoch+1) % args.log_metrics_interval == 0:
            #     if best_dice_score < train_metrics_agg[0]:
            #         save_path = f"{directory}/model-{args.model_name}-{args.classes}class-current_best_train.pt"
            #         torch.save({
            #                 'epoch': (epoch),
            #                 'model_state_dict': model.state_dict(),
            #                 'model_train_dice': train_metrics_agg[0],
            #                 'model_train_hd': train_metrics_agg[2],
            #                 'experiment_name': unique_experiment_name,
            #                 'experiment_key': experiment.get_key()
            #                 }, save_path)
            #         best_dice_score = train_metrics_agg[0]
            #         print(f"Current best train dice score {best_dice_score:.4f}. Model saved!")
                            
            # save best VALIDATION score
            # if (epoch+1) % args.validation_interval == 0:
            #     if best_dice_val_score < val_metrics_agg[0]:
            #         save_path = f"{directory}/model-{args.model_name}-{args.classes}class-current_best_val.pt"
            #         torch.save({
            #             'epoch': (epoch),
            #             'model_state_dict': model.state_dict(),
            #             'model_val_dice': val_metrics_agg[0],
            #             'model_val_hd': val_metrics_agg[2],
            #             'model_val_dice_multiclass': val_dice_multiclass_agg,
            #             'experiment_name': unique_experiment_name,
            #             'experiment_key': experiment.get_key()
            #             }, save_path)
            #         best_dice_val_score = val_metrics_agg[0]
            #         print(f"Current best binary segmentation validation dice score {best_dice_val_score:.4f}. Model saved!")
            #     if best_dice_multiclass_val_score < val_metrics_multiclass_agg:
            #         save_path = f"{directory}/model-{args.model_name}-{args.classes}class-current_best_multiclass_val.pt"
            #         torch.save({
            #             'epoch': (epoch),
            #             'model_state_dict': model.state_dict(),
            #             'model_val_dice': val_metrics_agg[0],
            #             'model_val_hd': val_metrics_agg[2],
            #             'model_val_dice_multiclass': val_dice_multiclass_agg,
            #             'experiment_name': unique_experiment_name,
            #             'experiment_key': experiment.get_key()
            #             }, save_path)
            #         best_dice_multiclass_val_score = val_metrics_multiclass_agg
            #         print(f"Current best multiclass segmentation validation dice score {best_dice_multiclass_val_score:.4f}. Model saved!")

            #save based on SAVE INTERVAL
            # if (epoch+1) % args.save_interval == 0 and epoch != 0:
            #     save_path = f"{directory}/model-{args.model_name}-{args.classes}class_val_{val_metrics_agg[0]:.4f}_train_{train_metrics_agg[0]:.4f}_epoch_{(epoch):04}.pt"
            #     #save based on optimizer save interval - allows to continue training
            #     if args.save_optimizer and epoch % args.save_optimiser_interval == 0 and epoch != 0:
            #         torch.save({
            #             'epoch': (epoch),
            #             'model_state_dict': model.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             'scheduler_state_dict' : scheduler.state_dict(),
            #             'model_train_dice': train_metrics_agg[0],
            #             'model_train_hd': train_metrics_agg[2],
            #             'model_val_dice': val_metrics_agg[0],
            #             'model_val_hd': val_metrics_agg[2],
            #             'experiment_name': unique_experiment_name,
            #             'experiment_key': experiment.get_key()
            #             }, save_path)
            #         print("Saved optimizer and scheduler state dictionaries.")
            #     else:
            #         torch.save({
            #             'epoch': (epoch),
            #             'model_state_dict': model.state_dict(),
            #             'model_train_dice': train_metrics_agg[0],
            #             'model_train_hd': train_metrics_agg[2],
            #             'model_val_dice': val_metrics_agg[0],
            #             'model_val_hd': val_metrics_agg[2],
            #             'experiment_name': unique_experiment_name,
            #             'experiment_key': experiment.get_key()
            #             }, save_path)
            #         print(f"Interval model saved! - train_dice: {train_metrics_agg[0]:.4f}, val_dice: {val_metrics_agg[0]:.4f}, best_val_dice: {best_dice_val_score:.4f}.")
        
        #Final epoch report
        # epoch_time=time.time() - start_time_epoch
        # print(f"Epoch: {epoch+1} finished. Total training loss: {train_loss_agg[0]:.4f} - total epoch time: {epoch_time:.2f}s.")
    
    print(f"Experiment finished! logging to comet server...")
    #wait to move logs to comet
    experiment.flush()
    experiment.end()
    print("---------------------------------------------------------\n")
    print (f"Experiment {unique_experiment_name} sent to server.")
    print("---------------------------------------------------------\n")

if __name__ == "__main__":
    # import torch.multiprocessing as mp
    # mp.set_start_method("spawn", force=True)
    main()