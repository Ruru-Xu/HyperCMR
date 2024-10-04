import pathlib
from argparse import ArgumentParser
from promptmr_task1 import PromptMR
from mri_data_task1 import *
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm import tqdm
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
import logging
import fastmri
from torchvision import models
from eagle_loss import Eagle_Loss

def log_to_tensorboard(writer, epoch, recons_pred, kspace_pred, target, fully_kspace, mode_name):
    # Compute the magnitude of the k-space data using fastmri
    kspace_magnitude = np.log(fastmri.complex_abs(kspace_pred).cpu().numpy() + 1e-16)

    # Ensure there are at least 2 elements in the required dimension
    if kspace_magnitude.shape[0] < 1:
        return

    # Select specific slices to visualize
    kspace_pred_image_0 = kspace_magnitude[0, 0, :, :]
    fully_kspace_magnitude = np.log(fastmri.complex_abs(fully_kspace).cpu().numpy() + 1e-16)
    fully_kspace_image_0 = fully_kspace_magnitude[0, 0, :, :]

    # Normalize for visualization
    kspace_pred_image_0 = (kspace_pred_image_0 - kspace_pred_image_0.min()) / (kspace_pred_image_0.max() - kspace_pred_image_0.min())
    fully_kspace_image_0 = (fully_kspace_image_0 - fully_kspace_image_0.min()) / (fully_kspace_image_0.max() - fully_kspace_image_0.min())

    # Convert to tensors and add channel dimension for visualization
    kspace_pred_image_0 = torch.tensor(kspace_pred_image_0).unsqueeze(0)
    fully_kspace_image_0 = torch.tensor(fully_kspace_image_0).unsqueeze(0)

    # Add images to TensorBoard
    writer.add_image(f'{mode_name}/kspace_pred_0', kspace_pred_image_0, epoch)
    writer.add_image(f'{mode_name}/fully_kspace_0', fully_kspace_image_0, epoch)

    # Visualize reconstructed images and target images
    recons_pred_image_0 = recons_pred[0, :, :].unsqueeze(0)
    target_image_0 = target[0, :, :].unsqueeze(0)

    # Normalize for visualization
    recons_pred_image_0 = (recons_pred_image_0 - recons_pred_image_0.min()) / (recons_pred_image_0.max() - recons_pred_image_0.min())
    target_image_0 = (target_image_0 - target_image_0.min()) / (target_image_0.max() - target_image_0.min())

    # Add images to TensorBoard
    writer.add_image(f'{mode_name}/recons_pred_0', recons_pred_image_0, epoch)
    writer.add_image(f'{mode_name}/target_0', target_image_0, epoch)

def create_dataloader(fileName, batch_size, args, mode, train=True):
    data_path = os.path.join(args.data_path, mode, args.h5py_folder)
    mask = FixedLowEquiSpacedMaskFunc(args.center_numbers, args.accelerations, allow_any_combination=True)
    if train:
        transform = PromptMrDataTransform(mask_func=mask, use_seed=False)
    else:
        transform = PromptMrDataTransform(mask_func=mask)

    data_module = CmrxReconDataModule(
        data_path=data_path,
        fileName=fileName,
        train_transform=transform if train else None,
        val_transform=transform if not train else None,
        batch_size=batch_size,
        num_workers=args.num_workers
    )

    dataset = data_module.train_dataloader().dataset if train else data_module.val_dataloader().dataset
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=train)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)

def normalize_kspace(kspace, epsilon=1e-8):
    kspace_magnitude = torch.sqrt(kspace[..., 0]**2 + kspace[..., 1]**2)
    max_val = kspace_magnitude.amax(dim=(-2, -3), keepdim=True)
    normalized_kspace = kspace_magnitude / (max_val + epsilon)
    return normalized_kspace

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, layers):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = torch.nn.ModuleList([vgg[i] for i in layers]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0.0
        for layer in self.vgg_layers:
            x = layer(x)
            y = layer(y)
            loss += F.mse_loss(x, y)
        return loss

def train_epoch(train_loader, model, optimizer, device, scaler, mode_name, epoch, writer):
    model.train()
    running_loss = 0.0
    vgg_loss = VGGPerceptualLoss(layers=[0, 5, 10, 19, 28]).to(device)
    eagle_loss = Eagle_Loss(patch_size=5)

    accumulation_steps = 8  # Number of minibatches to accumulate gradients over
    for i, batch in enumerate(tqdm(train_loader, desc=f"Training, {mode_name}")):
        fully_kspace, masked_kspace, mask, target = batch
        fully_kspace, masked_kspace, mask, target = fully_kspace.to(device), masked_kspace.to(device), mask.to(device), target.to(device)
        fully_kspace = torch.chunk(fully_kspace, 5, dim=1)[2]


        with autocast():
            recons_pred, kspace_pred = model(masked_kspace, mask)

            target_crop = crop_submission(target)
            recons_pred_crop = crop_submission(recons_pred)

            fully_kspace_normalized = normalize_kspace(fully_kspace)
            kspace_pred_normalized = normalize_kspace(kspace_pred)
            loss_fidelity = F.mse_loss(kspace_pred_normalized, fully_kspace_normalized)# Data Fidelity Loss (k-space domain)

            recons_pred_crop_norm = (recons_pred_crop - recons_pred_crop.amin(dim=(-1, -2), keepdim=True)) / (recons_pred_crop.amax(dim=(-1, -2), keepdim=True) - recons_pred_crop.amin(dim=(-1, -2), keepdim=True))
            target_crop_norm = (target_crop - target_crop.amin(dim=(-1, -2), keepdim=True)) / (target_crop.amax(dim=(-1, -2), keepdim=True) - target_crop.amin(dim=(-1, -2), keepdim=True))
            recons_pred_norm = (recons_pred - recons_pred.amin(dim=(-1, -2), keepdim=True)) / (recons_pred.amax(dim=(-1, -2), keepdim=True) - recons_pred.amin(dim=(-1, -2), keepdim=True))
            target_norm = (target - target.amin(dim=(-1, -2), keepdim=True)) / (target.amax(dim=(-1, -2), keepdim=True) - target.amin(dim=(-1, -2), keepdim=True))

            loss_eagle_cardiac = eagle_loss(recons_pred_crop_norm.unsqueeze(1), target_crop_norm.unsqueeze(1))
            loss_eagle_whole = eagle_loss(recons_pred_norm.unsqueeze(1), target_norm.unsqueeze(1))
            loss_recons_ssim = 1 - ssim(recons_pred_crop_norm.unsqueeze(1), target_crop_norm.unsqueeze(1), data_range=1.0)

            # Correctly prepare the input for VGG loss
            recons_pred_vgg = recons_pred_crop_norm.unsqueeze(1).expand(-1, 3, -1, -1)
            target_vgg = target_crop_norm.unsqueeze(1).expand(-1, 3, -1, -1)
            loss_vgg = vgg_loss(recons_pred_vgg, target_vgg)

            kspace_pred_norm = kspace_pred / (kspace_pred.shape[-3] * kspace_pred.shape[-2] * kspace_pred.shape[-1]) ** 0.5 # Normalize k-space for Regularization Loss
            loss_reg = torch.norm(kspace_pred_norm, p=1) + 0.05 * torch.norm(kspace_pred_norm, p=2)  # Regularization Loss

            total_loss = (1.0 * loss_fidelity +
                          1.0 * loss_recons_ssim +
                          0.05 * (0.6*loss_eagle_cardiac+0.4*loss_eagle_whole) +
                          0.01 * loss_reg +
                        0.01 * loss_vgg)
            total_loss = total_loss / accumulation_steps  # Normalize the loss

        scaler.scale(total_loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if torch.isnan(total_loss):
            raise ValueError("NaN detected in total_loss")
        running_loss += total_loss.item() * accumulation_steps

        writer.add_scalar(f'Train/{mode_name}/loss_fidelity', loss_fidelity.item(), epoch)
        writer.add_scalar(f'Train/{mode_name}/loss_recons_ssim', loss_recons_ssim.item(), epoch)
        writer.add_scalar(f'Train/{mode_name}/loss_hf', loss_eagle_cardiac.item(), epoch)
        writer.add_scalar(f'Train/{mode_name}/loss_reg', loss_reg.item(), epoch)
        writer.add_scalar(f'Train/{mode_name}/loss_vgg', loss_vgg.item(), epoch)
        writer.add_scalar(f'Train/{mode_name}/total_loss', total_loss.item(), epoch)

    return running_loss / len(train_loader)


def validate(val_loader, model, device, mode_name,  writer, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating, {mode_name}"):
            fully_kspace, masked_kspace, mask, target = batch
            fully_kspace, masked_kspace, mask, target = fully_kspace.to(device), masked_kspace.to(device), mask.to(
                device), target.to(device)
            fully_kspace = torch.chunk(fully_kspace, 5, dim=1)[2]

            with autocast():
                recons_pred, kspace_pred = model(masked_kspace, mask)

                target = crop_submission(target)
                recons_pred = crop_submission(recons_pred)

                target_data_range = target.amax(dim=(-1, -2), keepdim=True)
                recons_pred = (recons_pred - recons_pred.amin(dim=(-1, -2), keepdim=True)) / (
                            recons_pred.amax(dim=(-1, -2), keepdim=True) - recons_pred.amin(dim=(-1, -2), keepdim=True))
                target = (target - target.amin(dim=(-1, -2), keepdim=True)) / (
                            target_data_range - target.amin(dim=(-1, -2), keepdim=True))
                loss_recons_ssim = 1 - ssim(recons_pred.unsqueeze(1), target.unsqueeze(1), data_range=1.0)

            if torch.isnan(loss_recons_ssim):
                raise ValueError("NaN detected in total_loss")

            running_loss += loss_recons_ssim.item()

            writer.add_scalar(f'Train/{mode_name}/loss_recons_ssim', loss_recons_ssim.item(), epoch)

            # Log predictions to TensorBoard
            if epoch % args.log_every_n_epochs == 0:
                log_to_tensorboard(writer, epoch, recons_pred, kspace_pred, target, fully_kspace, mode_name)


        writer.add_scalar(f'Validation/{mode_name}_loss', running_loss / len(val_loader), epoch)
        return running_loss / len(val_loader)

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, f'{filename}')
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, f'best_model.pth.tar')
        torch.save(state, best_filepath)


def cli_main(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    pl.seed_everything(args.seed)
    # Set the environment variables manually
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29515'

    # Ensure only one process initializes the process group
    dist.init_process_group(backend='nccl', init_method='env://')

    # Set device for the current process
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')

    # Update rank and world_size in args
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()

    model = PromptMR(
        num_cascades=12,  # number of unrolled iterations
        num_adj_slices=5,  # number of adjacent slices

        n_feat0=48,  # number of top-level channels for PromptUnet
        feature_dim=[72, 96, 120],
        prompt_dim=[24, 48, 72],

        sens_n_feat0=24,
        sens_feature_dim=[36, 48, 60],
        sens_prompt_dim=[12, 24, 36],

        no_use_ca=False,
    ).to(device)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)



    # Set up logging to both file and console
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(args.experiments_output, 'train_acc10.log'))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    writer = SummaryWriter(log_dir=args.experiments_output)

    modality_names = ['aorta_sag', 'aorta_tra', 'cine_lax204', 'cine_lax168', 'cine_sax246', 'cine_sax162', 'cine_sax204', 'cine_lvot', 'T1map', 'T2map', 'tagging']

    train_loaders = {
        'aorta_sag': [create_dataloader('aorta_sag', args.aorta_batch_size, args, mode='Aorta')],
        'aorta_tra':[create_dataloader('aorta_tra', args.aorta_batch_size, args, mode='Aorta')],
        'cine_lax204': [create_dataloader('cine_lax204', args.cine_lax204_batch_size, args, mode='Cine')],
        'cine_lax168': [create_dataloader('cine_lax168', args.cine_lax168_batch_size, args, mode='Cine')],
        'cine_sax246': [create_dataloader('cine_sax246', args.cine_sax246_batch_size, args, mode='Cine')],
        'cine_sax162': [create_dataloader('cine_sax162', args.cine_sax162_batch_size, args, mode='Cine')],
        'cine_sax204': [create_dataloader('cine_sax204', args.cine_sax204_batch_size, args, mode='Cine')],
        'cine_lvot':   [create_dataloader('cine_lvot', args.cine_lvot_batch_size, args, mode='Cine')],
        'T1map':  [create_dataloader('T1map', args.t1map_batch_size, args, mode='Mapping')],
        'T2map':  [create_dataloader('T2map', args.t2map_batch_size, args, mode='Mapping')],
        'tagging': [create_dataloader('tagging', args.tagging_batch_size, args, mode='Tagging')],
    }

    val_loaders = {
        'aorta_sag': [create_dataloader('aorta_sag', args.aorta_batch_size, args, train=False, mode='Aorta')],
        'aorta_tra': [create_dataloader('aorta_tra', args.aorta_batch_size, args, train=False, mode='Aorta')],
        'cine_lax204': [create_dataloader('cine_lax204', args.cine_lax204_batch_size, args, train=False, mode='Cine')],
        'cine_lax168': [create_dataloader('cine_lax168', args.cine_lax168_batch_size, args, train=False, mode='Cine')],
        'cine_sax246': [create_dataloader('cine_sax246', args.cine_sax246_batch_size, args, train=False, mode='Cine')],
        'cine_sax162': [ create_dataloader('cine_sax162', args.cine_sax162_batch_size, args, train=False, mode='Cine')],
        'cine_sax204': [create_dataloader('cine_sax204', args.cine_sax204_batch_size, args, train=False, mode='Cine')],
        'cine_lvot':   [create_dataloader('cine_lvot', args.cine_lvot_batch_size, args, train=False, mode='Cine')],
        'T1map':  [create_dataloader('T1map', args.t1map_batch_size, args, train=False, mode='Mapping')],
        'T2map': [create_dataloader('T2map', args.t2map_batch_size, args, train=False, mode='Mapping')],
        'tagging': [create_dataloader('tagging', args.tagging_batch_size, args, train=False, mode='Tagging')],
    }

    logging.info(f'Training slices: \n aorta_sag:{len(train_loaders["aorta_sag"][0].dataset)}, aorta_tra:{len(train_loaders["aorta_tra"][0].dataset)},'
                 f'cine_lax204:{sum(len(loader.dataset) for loader in train_loaders["cine_lax204"])}, '
                 f'cine_lax168:{sum(len(loader.dataset) for loader in train_loaders["cine_lax168"])}, '
                 f'cine_sax246:{sum(len(loader.dataset) for loader in train_loaders["cine_sax246"])}, '
                 f'cine_sax162:{sum(len(loader.dataset) for loader in train_loaders["cine_sax162"])}, '
                 f'cine_sax204:{sum(len(loader.dataset) for loader in train_loaders["cine_sax204"])}, '
                 f'cine_lvot:{len(train_loaders["cine_lvot"][0].dataset)}, '
                 f'T1map:{len(train_loaders["T1map"][0].dataset)}, T2map:{len(train_loaders["T2map"][0].dataset)}, '
                 f'tagging:{len(train_loaders["tagging"][0].dataset)}')
    logging.info(f'Validating slices: \n aorta_sag:{len(val_loaders["aorta_sag"][0].dataset)}, aorta_tra:{len(val_loaders["aorta_tra"][0].dataset)},'
                 f'cine_lax204:{sum(len(loader.dataset) for loader in val_loaders["cine_lax204"])}, '
                 f'cine_lax168:{sum(len(loader.dataset) for loader in val_loaders["cine_lax168"])}, '
                 f'cine_sax246:{sum(len(loader.dataset) for loader in val_loaders["cine_sax246"])}, '
                 f'cine_sax162:{sum(len(loader.dataset) for loader in val_loaders["cine_sax162"])}, '
                 f'cine_sax204:{sum(len(loader.dataset) for loader in val_loaders["cine_sax204"])}, '
                 f'cine_lvot:{len(val_loaders["cine_lvot"][0].dataset)}, '
                 f'T1map:{len(val_loaders["T1map"][0].dataset)}, T2map:{len(val_loaders["T2map"][0].dataset)}, '
                 f'tagging:{len(val_loaders["tagging"][0].dataset)}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scaler = GradScaler()

    total_training_time = 0
    epoch_count = 0

    epochs_no_improve = 0
    best_val_loss = float('inf')

    for epoch in range(args.max_epochs):
        epoch_start_time = time.time()
        total_train_loss = 0.0
        total_val_loss = 0.0

        for modality in modality_names:
            modality_train_losses = []
            modality_val_losses = []

            for train_loader in train_loaders[modality]:
                train_loss = train_epoch(train_loader, model, optimizer, args.device, scaler, modality, epoch, writer)
                modality_train_losses.append(train_loss)

            for val_loader in val_loaders[modality]:
                val_loss = validate(val_loader, model, args.device, modality, writer, epoch)
                modality_val_losses.append(val_loss)

            modality_train_loss = sum(modality_train_losses) / len(modality_train_losses)
            modality_val_loss = sum(modality_val_losses) / len(modality_val_losses)

            total_train_loss += modality_train_loss
            total_val_loss += modality_val_loss

            logger.info(f"Modality {modality}, Train Loss: {modality_train_loss}, Validation Loss: {modality_val_loss}")

        epoch_train_loss = total_train_loss / len(modality_names)
        epoch_val_loss = total_val_loss / len(modality_names)
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        epoch_count += 1
        logger.info(
            f"Epoch {epoch_count}, Train Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}, Time: {epoch_time:.2f} seconds")

        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        optimizer.step()
        scheduler.step()

        save_checkpoint({
            'epoch': epoch_count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss
        }, is_best, args.experiments_output, filename=f'checkpoint_epoch_{epoch_count}.pth.tar')

        if epochs_no_improve >= 1:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        logging.info(f"Total Training Time after {epoch_count} epochs: {total_training_time:.2f} seconds")


def build_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", default=pathlib.Path(
        '/media/ruru/dd2c0fe5-b971-4e64-a050-e13627f23931/work/MICCAI/miccai2024/MICCAIChallenge2024/ChallengeData/MultiCoil'))
    parser.add_argument("--experiments_output",
                        default=pathlib.Path('/home/ruru/Documents/work/MICCAI2024/task1/output/'))
    parser.add_argument("--aorta_batch_size", default=2, type=int)
    parser.add_argument("--cine_lax204_batch_size", default=2, type=int)
    parser.add_argument("--cine_lax168_batch_size", default=2, type=int)
    parser.add_argument("--cine_sax162_batch_size", default=2, type=int)
    parser.add_argument("--cine_sax204_batch_size", default=1, type=int)
    parser.add_argument("--cine_sax246_batch_size", default=1, type=int)
    parser.add_argument("--cine_lvot_batch_size", default=2, type=int)
    parser.add_argument("--t1map_batch_size", default=2, type=int)
    parser.add_argument("--t2map_batch_size", default=4, type=int)
    parser.add_argument("--tagging_batch_size", default=2, type=int)
    parser.add_argument("--mode", default="train", choices=("train"), type=str, help="Operation mode")
    parser.add_argument("--gpus", default=4, type=int, help="Number of GPUs to use")
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--center_numbers", nargs="+", default=[16], type=int,
                        help="Number of center lines to use in mask")
    parser.add_argument("--accelerations", nargs="+", default=[10], type=int, help="Acceleration rates to use for masks")
    parser.add_argument("--h5py_folder", default='TrainingSet/h5_FullSample', type=str,
                        help="folder name for converted h5py files")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of workers to use in data loader")

    # training params (opt)
    parser.add_argument("--lr", default=0.0001, type=float, help="Adam learning rate")
    parser.add_argument("--lr_step_size", default=3, type=int, help="Epoch at which to decrease step size")
    parser.add_argument("--lr_gamma", default=0.9, type=float, help="Extent to which step size should be decreased")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Strength of weight decay regularization")
    parser.add_argument("--use_checkpoint", default=False, help="Use checkpoint (default: False)")
    parser.add_argument("--low_mem", action="store_true",
                        help="consume less memory by computing sens_map coil by coil (default: False)")
    parser.add_argument("--log_every_n_epochs", default=1, type=int, help="Log outputs to TensorBoard every N epochs")

    # trainer config
    parser.add_argument("--seed", default=42, help="random seed")
    parser.add_argument("--deterministic", default=False, help="makes things slower, but deterministic")
    parser.add_argument("--max_epochs", default=50, type=int, help="max number of epochs")
    parser.add_argument("--gradient_clip_val", default=0.01, type=float, help="value for gradient clipping")
    parser.add_argument("--accumulate_grad_batches", default=1, type=int,
                        help="Accumulate gradients over N batches before updating weights")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv('LOCAL_RANK', 0)))

    args = parser.parse_args()
    acc_folder = "acc_" + "_".join(map(str, args.accelerations))
    args.experiments_output = args.experiments_output / acc_folder
    if not args.experiments_output.exists():
        args.experiments_output.mkdir(parents=True)

    return args


if __name__ == "__main__":
    args = build_args()
    cli_main(args)


