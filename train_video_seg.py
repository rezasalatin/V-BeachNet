import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import DataLoader

from video_module.dataset import Water_Image_Train_DS
from video_module.model import AFB_URR, FeatureBank
import myutils

# Enable CUDA launch blocking for debugging; set to '0' to disable.
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def get_args():
    parser = argparse.ArgumentParser(description='Train V-BeachNet')
    parser.add_argument('--gpu', type=int, default=0, help='GPU card id.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset folder.')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed.')
    parser.add_argument('--log', action='store_true', help='Save the training results.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (default: 1e-5).')
    parser.add_argument('--lu', type=float, default=0.5, help='Regularization factor (default: 0.5).')
    parser.add_argument('--resume', type=str, help='Path to checkpoint (default: none).')
    parser.add_argument('--new', action='store_true', help='Train the model from scratch.')
    parser.add_argument('--scheduler-step', type=int, default=25, help='Scheduler step size (default: 25).')
    parser.add_argument('--total-epochs', type=int, default=100, help='Total number of epochs (default: 100).')
    parser.add_argument('--budget', type=int, default=300000, help='Maximum number of features in the feature bank (default: 300000).')
    parser.add_argument('--obj-n', type=int, default=2, help='Maximum number of objects trained simultaneously.')
    parser.add_argument('--clip-n', type=int, default=6, help='Maximum number of frames in a batch.')

    return parser.parse_args()

def train_model(model, dataloader, criterion, optimizer):
    stats = myutils.AvgMeter()
    uncertainty_stats = myutils.AvgMeter()
    progress_bar = tqdm(dataloader)
    
    for _, sample in enumerate(progress_bar):
        frames, masks, obj_n, info = sample

        if obj_n.item() == 1:
            continue

        frames, masks = frames[0].to(device), masks[0].to(device)
        fb_global = FeatureBank(obj_n.item(), args.budget, device)
        k4_list, v4_list = model.memorize(frames[:1], masks[:1])
        fb_global.init_bank(k4_list, v4_list)

        scores, uncertainty = model.segment(frames[1:], fb_global)
        label = torch.argmax(masks[1:], dim=1).long()

        optimizer.zero_grad()
        loss = criterion(scores, label) + args.lu * uncertainty
        loss.backward()
        optimizer.step()

        uncertainty_stats.update(uncertainty.item())
        stats.update(loss.item())
        progress_bar.set_postfix(loss=f'{loss.item():.5f} (Avg: {stats.avg:.5f}, Uncertainty Avg: {uncertainty_stats.avg:.5f})')

    return stats.avg

def main():
    dataset = Water_Image_Train_DS(root=args.dataset, output_size=400, clip_n=args.clip_n, max_obj_n=args.obj_n)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    print(myutils.gct(), f'Dataset with {len(dataset)} training cases.')

    model = AFB_URR(device, update_bank=False, load_imagenet_params=True).to(device)
    model.train()
    model.apply(myutils.set_bn_eval)

    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), args.lr)

    start_epoch, best_loss = 0, float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'], strict=False)
            seed = checkpoint.get('seed', int(time.time()))

            if not args.new:
                start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                best_loss = checkpoint['loss']
                print(myutils.gct(), f'Resumed from checkpoint {args.resume} (Epoch: {start_epoch-1}, Best Loss: {best_loss}).')
            else:
                print(myutils.gct(), f'Loaded checkpoint {args.resume}. Training from scratch.')
        else:
            raise FileNotFoundError(f'No checkpoint found at {args.resume}')
    else:
        seed = args.seed if args.seed >= 0 else int(time.time())

    print(myutils.gct(), 'Random seed:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.5, last_epoch=start_epoch-1)

    for epoch in range(start_epoch, args.total_epochs):
        print(f'\n{myutils.gct()} Epoch: {epoch}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        loss = train_model(model, dataloader, criterion, optimizer)

        if args.log:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'seed': seed
            }
            torch.save(checkpoint, os.path.join(model_path, 'final.pth'))
            if loss < best_loss:
                best_loss = loss
                torch.save(checkpoint, os.path.join(model_path, 'best.pth'))
                print('Updated best model.')

        scheduler.step()

if __name__ == '__main__':
    args = get_args()
    print(myutils.gct(), f'Arguments: {args}')

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', args.gpu)
    else:
        raise ValueError('CUDA is required. Ensure --gpu is set to >= 0.')

    if args.log:
        log_dir = os.path.join('logs', time.strftime('%Y%m%d-%H%M%S'))
        model_path = os.path.join(log_dir, 'model')
        os.makedirs(model_path, exist_ok=True)
        myutils.save_scripts(log_dir, scripts_to_save=glob('*.*'))
        myutils.save_scripts(log_dir, scripts_to_save=glob('dataset/*.py', recursive=True))
        myutils.save_scripts(log_dir, scripts_to_save=glob('model/*.py', recursive=True))
        myutils.save_scripts(log_dir, scripts_to_save=glob('myutils/*.py', recursive=True))
        print(myutils.gct(), f'Created log directory: {log_dir}')

    main()
    print(myutils.gct(), 'Training completed.')
