import os
import yaml
import argparse
# ~/miniconda3/envs/dev3d2/lib/python3.9/site-packages/
#load experiment config
config_file = 'config.yaml'
with open(config_file, 'r') as file:
    general_config = yaml.safe_load(file)
    
#command line arguments - update config.yaml 
if general_config['general']['config_source'] == 'cmd':
    print(" *** Applying configuration from commandline experiment parameters ***")
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cuda_device_id', default=1, type=int, choices=[0,1])
    parser.add_argument('--seg_loss_name', default='GWD', type=str, choices=['DiceCELoss', 'DiceFocalLoss', 'GDL', 'GWD'])
    parser.add_argument('--classification_loss', default='cross_entropy', type=str, choices=['cross_entropy', 'focal_loss'])
    parser.add_argument('--is_weighted_cls', default=True, action='store_true')
    parser.add_argument('--weighting_mode', default='default', type=str, choices=['default', 'GDL'])
    parser.add_argument('--classes', default=33, type=int, choices=[9, 17, 33])
    
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
import itertools
import json
import re
import glob
import random
import time
import warnings
import numpy as np
from datetime import datetime

# TORCH modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler

#MONAI modules
from monai.losses import DiceLoss
from monai.metrics import MSEMetric
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric, DiceMetric
from monai.metrics import CumulativeAverage
from monai.optimizers import WarmupCosineSchedule
from monai.data import ThreadDataLoader, DataLoader, decollate_batch
from monai.data.dataset import PersistentDataset, Dataset
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from PIL import Image

from src.cuda_setup import configure_cuda
from src.data_preparation import split_train_val, create_domain_labels, build_sampler, load_dataset_json
from src.model import DWNet
from src.scheduler import CosineAnnealingWarmupRestarts
from src.losses import DiceCELoss, DiceFocalLoss
from src.transforms import Transforms
from src.logger import DummyExperiment

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
    autocast_d_type = torch.float32 
    if args.use_scaler:
        TORCH_DTYPES = {
        'bfloat16': torch.bfloat16,    
        'float16': torch.float16,     
        'float32': torch.float32
        }
        scaler = torch.cuda.amp.GradScaler()
        autocast_d_type=TORCH_DTYPES[args.autocast_dtype]
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
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    for subdir in ['train', 'val', 'test']:
        subdir_path = os.path.join(args.cache_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
                        
    if args.clear_cache:
        print("Clearing cache...")
        train_cache = glob.glob(os.path.join(args.cache_dir, 'train/*.pt'))
        val_cache = glob.glob(os.path.join(args.cache_dir, 'val/*.pt'))
        test_cache = glob.glob(os.path.join(args.cache_dir, 'test/*.pt'))
        if len(train_cache) != 0:
            for file in train_cache:
                os.remove(file)
        if len(val_cache) != 0:
            for file in val_cache:
                os.remove(file)
        if len(test_cache) != 0:
            for file in test_cache:
                os.remove(file)
        print(f"Cleared cache in dir: {args.cache_dir}, train: {len(train_cache)} files, val: {len(val_cache)} files, test: {len(test_cache)} files.")

    return scaler, autocast_d_type, device

def training_step(args, loss_fn, batch_idx, epoch, model, optimizer, scaler, data_sample, train_loader, logger, experiment, autocast_d_type, device):
    
    #OPTIMIZATION
    print(f"batch size: {args.batch_size}, batch_idx: {batch_idx}, {data_sample['image'].shape}, {data_sample['label'].shape}, source: {data_sample['image'].meta['filename_or_obj']}.")
    # with torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type):
    #     output = model(data_sample["image"])
    #     loss = loss_fn(output, data_sample["label"])
    
    # if args.use_scaler and scaler is not None:
    #     scaler.scale(loss).backward()

    #     if args.grad_clip:
    #         scaler.unscale_(optimizer)
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2.0)

    #     scaler.step(optimizer)
    #     scaler.update()
    # else:
    #     loss.backward()

    #     if args.grad_clip:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2.0)

    #     optimizer.step()
        
    # optimizer.zero_grad()
    
    #get predictions
    #log metrics
    #log qualitative results
    
# INFERENCE STEP
def inference_step(args, loss_fn, batch_idx, epoch, model, optimizer, scaler, data_sample, train_loader, experiment, autocast_d_type, device):
    with warnings.catch_warnings(), \
         torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type):
        output = sliding_window_inference(data_sample["image"], roi_size=args.patch_size, sw_batch_size=8, predictor=model, 
                                            overlap=0.5, sw_device=device, device=device, mode='gaussian', sigma_scale=0.125,
                                            padding_mode='constant', cval=0, progress=False)
    #get predictions
    #log metrics
    #log qualitative results

def main():
    #load experiment configurations, setup cuda
    scaler, autocast_d_type, device = setup_training(args)
    
    #setup experiment and logger
    if args.comet:
        experiment = Experiment(project_name="tf3")
        unique_experiment_name = experiment.get_name()
        experiment_key = experiment.get_key()
    else:
        experiment = DummyExperiment()
        unique_experiment_name = uuid.uuid4().hex
        args.batch_size=1
        args.validation_interval = 5
        args.log_batch_interval = 5
        args.log_metrics_interval = 5
        args.multiclass_metrics_interval = 5
        args.multiclass_metrics_epoch = 5
        args.log_slice_interval = 1
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

    train_data, val_data = split_train_val(args)
    train_labels = create_domain_labels(train_data)
    val_labels = create_domain_labels(val_data)

    trans=Transforms(args, device=device)

    if args.use_persistent_dataset:
        train_dataset = PersistentDataset(data=train_data, transform=trans.train_transform, cache_dir=os.path.join(args.cache_dir, 'train'))
        val_dataset = PersistentDataset(data=val_data, transform=trans.inference_transform, cache_dir=os.path.join(args.cache_dir, 'val'))
    else:
        # Your normal Dataset class expects list of filenames or similar
        train_dataset = Dataset(train_data, transform=trans.train_transform, root_dir=args.data)
        val_dataset = Dataset(val_data, transform=trans.inference_transform, root_dir=args.data)

    train_sampler = build_sampler(train_dataset, train_labels, args)
    val_sampler = build_sampler(val_dataset, val_labels, args)
        
    if args.use_thread_loader:
        train_loader = ThreadDataLoader(train_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = ThreadDataLoader(val_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=val_sampler)
        # test_loader_A = ThreadDataLoader(test_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=list(range(0,11)))
        # test_loader_B = ThreadDataLoader(test_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=list(range(11,20)))
    else:
        train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, sampler=train_sampler, worker_init_fn=np.random.seed(args.seed))
        val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size_val, sampler=val_sampler)
        # test_loader_A = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size_val, sampler=list(range(0,11)))
        # test_loader_B = DataLoader(test_dataset, num_workers=args.num_workers,batch_size=args.batch_size_val, sampler=list(range(11,20)))
    
    #setup model, optimizer, scheduler, losses, metrics
    #MODEL
    #num classes = 10 anatomical classes + 32 tooth classes + 3 canal classes + 1 pulp
    model = DWNet(spatial_dims=3, in_channels=1, out_channels=46, act=args.activation, norm=args.norm,
                  bias=False, backbone_name=args.backbone_name, configuration='UNET')
    if args.parallel:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    #LOSSES
    if args.weighting_mode != 'none':
        #TODO implement weights calculation, maybe json file?
        if args.weighting_mode == 'inverse_frequency_class_weights':
            weights = torch.tensor(class_labels['weights'], dtype=torch.float32, device=device)
        else:
            raise NotImplementedError(f"Weighting mode {args.weighting_mode} not implemented.")
    else:
        # no weights
        weights = None
        
    weights = None
    if args.seg_loss_name=="DiceCELoss":
        criterion_seg = DiceCELoss(include_background=args.include_background_loss, ce_weight=weights, to_onehot_y=True, softmax=True)
    elif args.seg_loss_name=="FocalDice":
        criterion_seg = DiceFocalLoss(include_background=args.include_background_loss, focal_weight=weights, to_onehot_y=True, softmax=True, gamma=args.wasserstein_distance_matrix)
    
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
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs+100)
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
    reduction='mean_batch'
    seg_metrics = [
        DiceMetric(include_background=args.include_background_metrics, reduction=reduction),
        SurfaceDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', reduction=reduction),
        HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean',
                                percentile=None, get_not_nans=False, directed=False, reduction=reduction),
    ]
    seg_metrics_multiclass = [
        DiceMetric(include_background=args.include_background_metrics, reduction=reduction, ignore_empty=True),
        SurfaceDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', reduction=reduction),
        HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean',
                                percentile=None, get_not_nans=False, directed=False, reduction=reduction),
    ]
    seg_metrics_binary = [
        DiceMetric(include_background=args.include_background_metrics, reduction='none'),
        HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean',
                                get_not_nans=False, directed=False, reduction='none')
    ]
    
    edt_reg_metrics=[MSEMetric(reduction=reduction)] 

    #train loss
    train_loss_cum = CumulativeAverage()
    edt_loss_cum = CumulativeAverage()
    seed_loss_cum = CumulativeAverage()
    seg_loss_cum = CumulativeAverage()
    seg_mlt_loss_cum = CumulativeAverage()
    angle_loss_cum = CumulativeAverage()
    training_loss_cms = [train_loss_cum, edt_loss_cum, seed_loss_cum, seg_loss_cum, seg_mlt_loss_cum, angle_loss_cum]
    
    #train metrics
    #binary
    train_dice_cum = CumulativeAverage()
    train_assd_cum = CumulativeAverage()
    train_hd_cum = CumulativeAverage()
    train_mse_edt_cum = CumulativeAverage()
    #multiclass
    train_dice_multiclass_cum = CumulativeAverage()
    train_assd_multiclass_cum = CumulativeAverage()
    train_hd_multiclass_cum = CumulativeAverage()
    training_metrics_cms = [train_dice_cum, train_assd_cum, train_hd_cum, train_mse_edt_cum]
    training_metrics_mlt_cms = [train_dice_multiclass_cum, train_assd_multiclass_cum, train_hd_multiclass_cum]
    #val metrics
    #binary
    val_dice_cum = CumulativeAverage()
    val_assd_cum = CumulativeAverage()
    val_hd_cum = CumulativeAverage()
    train_assd_cum = CumulativeAverage()
    val_mse_edt_cum = CumulativeAverage()
    #multiclass
    val_dice_multiclass_cum = CumulativeAverage()
    val_assd_multiclass_cum = CumulativeAverage()
    val_hd_multiclass_cum = CumulativeAverage()
    val_metrics_cms = [val_dice_cum, val_assd_cum, val_hd_cum, val_mse_edt_cum]
    val_metrics_mlt_cms = [val_dice_multiclass_cum, val_assd_multiclass_cum, val_hd_multiclass_cum]
        
    best_dice_score = 0.0
    best_dice_val_score = 0.0
    best_dice_multiclass_val_score = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        print(f"Starting epoch {epoch + 1}")
        model.train()
        #TRAINING LOOP
        start_time_epoch = time.time()
        for batch_idx, train_data in enumerate(train_loader):
            training_step(args, criterion_seg, batch_idx, epoch, model, optimizer, scaler, train_data, train_loader, 
                          logger, experiment, autocast_d_type, device)
        epoch_time=time.time() - start_time_epoch
        print(f" Train loop finished - total time: {epoch_time:.2f}s.")

        #RESET METRICS after training step
        if args.classes > 1:
            _ = [func.reset() for func in seg_metrics]
            _ = [func.reset() for func in edt_reg_metrics]
            if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                _ = [func.reset() for func in seg_metrics_multiclass]
        else:
            _ = [func.reset() for func in seg_metrics_binary]

        #VALIDATION
        model.eval()
        if (epoch+1) % args.validation_interval == 0 and epoch != 0:
            print("Starting validation...")
            start_time_validation = time.time()
            for batch_idx, val_data in enumerate(val_loader):
                with torch.no_grad():
                    inference_step(args, batch_idx, epoch, model, val_data, val_loader, log, experiment,
                                seg_metrics_binary, val_dice_cum, val_hd_cum, autocast_d_type, device)
            val_time=time.time() - start_time_validation
            print( f"Validation time: {val_time:.2f}s")
                    
            #RESET METRICS after validation step
            if args.classes > 1:
                _ = [func.reset() for func in seg_metrics]
                _ = [func.reset() for func in edt_reg_metrics]
                if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                    _ = [func.reset() for func in seg_metrics_multiclass]
            else:
                _ = [func.reset() for func in seg_metrics_binary]

            #aggregate metrics after validation step
            val_metrics_agg = [cum.aggregate() for cum in val_metrics_cms]
            val_metrics_multiclass_agg = val_dice_multiclass_cum.aggregate()
            
            #AGGREGATE RUNNING AVERAGES
            train_loss_agg = [cum.aggregate() for cum in training_loss_cms]
            if (epoch+1) % args.log_metrics_interval == 0:
                train_metrics_agg = [cum.aggregate() for cum in training_metrics_cms]
            if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                train_dice_multiclass_agg = train_dice_multiclass_cum.aggregate()
                train_dice_multiclass_cum.reset()
                experiment.log_metric("train_dice_multiclass", train_dice_multiclass_agg, epoch=epoch)
                if (epoch+1) % args.validation_interval == 0:
                    val_dice_multiclass_agg = val_dice_multiclass_cum.aggregate()
                    val_dice_multiclass_cum.reset()
                    experiment.log_metric("val_dice_multiclass", val_dice_multiclass_agg, epoch=epoch)
            else:
                train_dice_multiclass_agg = 0.0
                val_dice_multiclass_agg = 0.0
                    
            #reset running averages
            _ = [cum.reset() for cum in training_loss_cms]
            _ = [cum.reset() for cum in training_metrics_cms]
            if (epoch+1) % args.validation_interval == 0:
                _ = [cum.reset() for cum in val_metrics_cms]
            
            #TEST
            if args.perform_test:
                for batch_idx, test_sample in enumerate(test_loader):
                    with torch.no_grad():
                        inference_step(args, batch_idx, epoch, model, test_sample, test_loader, log, experiment,
                                        seg_metrics_binary, val_dice_cum, val_hd_cum, autocast_d_type, device)

        scheduler.step()
        #LOG METRICS TO COMET
        # CHECKPOINTS SAVE
        if args.save_checkpoints:
            #create unique experiment name
            directory = f"checkpoints/{args.checkpoint_dir}/{unique_experiment_name}/classes_{str(args.classes)}_{loss_name}"
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            # save best TRAIN model
            if (epoch+1) % args.log_metrics_interval == 0:
                if best_dice_score < train_metrics_agg[0]:
                    save_path = f"{directory}/model-{args.model_name}-{args.classes}class-current_best_train.pt"
                    torch.save({
                            'epoch': (epoch),
                            'model_state_dict': model.state_dict(),
                            'model_train_dice': train_metrics_agg[0],
                            'model_train_hd': train_metrics_agg[2],
                            'experiment_name': unique_experiment_name,
                            'experiment_key': experiment.get_key()
                            }, save_path)
                    best_dice_score = train_metrics_agg[0]
                    print(f"Current best train dice score {best_dice_score:.4f}. Model saved!")
                            
            # save best VALIDATION score
            if (epoch+1) % args.validation_interval == 0:
                if best_dice_val_score < val_metrics_agg[0]:
                    save_path = f"{directory}/model-{args.model_name}-{args.classes}class-current_best_val.pt"
                    torch.save({
                        'epoch': (epoch),
                        'model_state_dict': model.state_dict(),
                        'model_val_dice': val_metrics_agg[0],
                        'model_val_hd': val_metrics_agg[2],
                        'model_val_dice_multiclass': val_dice_multiclass_agg,
                        'experiment_name': unique_experiment_name,
                        'experiment_key': experiment.get_key()
                        }, save_path)
                    best_dice_val_score = val_metrics_agg[0]
                    print(f"Current best binary segmentation validation dice score {best_dice_val_score:.4f}. Model saved!")
                if best_dice_multiclass_val_score < val_metrics_multiclass_agg:
                    save_path = f"{directory}/model-{args.model_name}-{args.classes}class-current_best_multiclass_val.pt"
                    torch.save({
                        'epoch': (epoch),
                        'model_state_dict': model.state_dict(),
                        'model_val_dice': val_metrics_agg[0],
                        'model_val_hd': val_metrics_agg[2],
                        'model_val_dice_multiclass': val_dice_multiclass_agg,
                        'experiment_name': unique_experiment_name,
                        'experiment_key': experiment.get_key()
                        }, save_path)
                    best_dice_multiclass_val_score = val_metrics_multiclass_agg
                    print(f"Current best multiclass segmentation validation dice score {best_dice_multiclass_val_score:.4f}. Model saved!")

            #save based on SAVE INTERVAL
            if (epoch+1) % args.save_interval == 0 and epoch != 0:
                save_path = f"{directory}/model-{args.model_name}-{args.classes}class_val_{val_metrics_agg[0]:.4f}_train_{train_metrics_agg[0]:.4f}_epoch_{(epoch):04}.pt"
                #save based on optimizer save interval - allows to continue training
                if args.save_optimizer and epoch % args.save_optimiser_interval == 0 and epoch != 0:
                    torch.save({
                        'epoch': (epoch),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict' : scheduler.state_dict(),
                        'model_train_dice': train_metrics_agg[0],
                        'model_train_hd': train_metrics_agg[2],
                        'model_val_dice': val_metrics_agg[0],
                        'model_val_hd': val_metrics_agg[2],
                        'experiment_name': unique_experiment_name,
                        'experiment_key': experiment.get_key()
                        }, save_path)
                    print("Saved optimizer and scheduler state dictionaries.")
                else:
                    torch.save({
                        'epoch': (epoch),
                        'model_state_dict': model.state_dict(),
                        'model_train_dice': train_metrics_agg[0],
                        'model_train_hd': train_metrics_agg[2],
                        'model_val_dice': val_metrics_agg[0],
                        'model_val_hd': val_metrics_agg[2],
                        'experiment_name': unique_experiment_name,
                        'experiment_key': experiment.get_key()
                        }, save_path)
                    print(f"Interval model saved! - train_dice: {train_metrics_agg[0]:.4f}, val_dice: {val_metrics_agg[0]:.4f}, best_val_dice: {best_dice_val_score:.4f}.")
        
        #Final epoch report
        epoch_time=time.time() - start_time_epoch
        print(f"Epoch: {epoch+1} finished. Total training loss: {train_loss_agg[0]:.4f} - total epoch time: {epoch_time:.2f}s.")
    
    print(f"Experiment finished! logging to comet server...")
    #wait to move logs to comet
    experiment.flush()
    experiment.end()
    print("---------------------------------------------------------\n")
    print (f"Experiment {unique_experiment_name} sent to server.")
    print("---------------------------------------------------------\n")

if __name__ == "__main__":
    main()