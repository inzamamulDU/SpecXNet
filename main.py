import argparse
import os
import random
import shutil
import time
import warnings
import sys
import csv
from glob import glob
from typing import List, Tuple

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # allow truncated JPEG/PNG to load
Image.MAX_IMAGE_PIXELS = None            # (optional) disable DecompressionBomb checks

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data import random_split, Dataset
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tensorboardX import SummaryWriter
import numpy as np

import model_zoo

class NullWriter:
    def add_scalar(self, *args, **kwargs): 
        return
    def close(self): 
        return

# -----------------------------
# Helpers for CSV-based loading
# -----------------------------
_CANONICAL_CLASSES = ['fake', 'real']
_CLASS_TO_IDX = {'fake': 0, 'real': 1}
IMG_EXTS_DEFAULT = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

def _read_dir_csv(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Reads a CSV with optional headers containing 'real' and 'fake' columns.
    Each row can fill one or both columns. Empty cells allowed.
    If no header, assumes col0=real and col1=fake if present.
    Only returns directories that actually exist.
    """
    real_dirs, fake_dirs = [], []
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        sample = f.read(2048)
        f.seek(0)
        sniffer = csv.Sniffer()
        try:
            has_header = sniffer.has_header(sample)
        except csv.Error:
            has_header = False

        reader = csv.reader(f)
        if has_header:
            header = next(reader, [])
            lower = [h.strip().lower() for h in header]
            idx_real = None
            idx_fake = None
            for i, name in enumerate(lower):
                if name in ('real', 'reals', 'real_dir', 'real_dirs'):
                    idx_real = i
                if name in ('fake', 'fakes', 'fake_dir', 'fake_dirs'):
                    idx_fake = i
            if idx_real is None and len(lower) >= 1:
                idx_real = 0
            if idx_fake is None and len(lower) >= 2:
                idx_fake = 1

            for row in reader:
                if not row:
                    continue
                if idx_real is not None and idx_real < len(row):
                    v = row[idx_real].strip()
                    if v:
                        real_dirs.append(v)
                if idx_fake is not None and idx_fake < len(row):
                    v = row[idx_fake].strip()
                    if v:
                        fake_dirs.append(v)
        else:
            # no header: assume col0=real, col1=fake (if exists)
            for row in reader:
                if not row:
                    continue
                if len(row) >= 1 and row[0].strip():
                    real_dirs.append(row[0].strip())
                if len(row) >= 2 and row[1].strip():
                    fake_dirs.append(row[1].strip())

    real_dirs = [d for d in sorted(set(real_dirs)) if os.path.isdir(d)]
    fake_dirs = [d for d in sorted(set(fake_dirs)) if os.path.isdir(d)]
    if len(real_dirs) == 0 and len(fake_dirs) == 0:
        raise RuntimeError(f"No valid directories found in CSV: {csv_path}")
    return real_dirs, fake_dirs


def _list_images_in_dirs(dirs: List[str], exts: List[str]) -> List[str]:
    files = []
    exts_lower = {e.lower() for e in exts}
    for d in dirs:
        for ext in exts_lower:
            files.extend(glob(os.path.join(d, f'**/*{ext}'), recursive=True))
            files.extend(glob(os.path.join(d, f'**/*{ext.upper()}'), recursive=True))
    files = [p for p in files if os.path.isfile(p)]
    return files


class FolderListImageDataset(Dataset):
    """
    Dataset built from directory lists for 'fake' and 'real' (labels fake=0, real=1).
    Scans the directories (recursively), assigns labels, and returns (image_tensor, label).
    Robust to truncated/corrupt files.
    """
    def __init__(
        self,
        fake_dirs: List[str],
        real_dirs: List[str],
        exts: List[str],
        transform=None,
        max_bad_retries: int = 5,
        fallback_size: int = 299,
        use_opencv: bool = False,
    ):
        self.transform = transform
        self.max_bad_retries = max_bad_retries
        self.fallback_size = fallback_size
        self.use_opencv = use_opencv

        fake_files = _list_images_in_dirs(fake_dirs, exts) if fake_dirs else []
        real_files = _list_images_in_dirs(real_dirs, exts) if real_dirs else []
        self.samples = [(p, _CLASS_TO_IDX['fake']) for p in fake_files] + \
                       [(p, _CLASS_TO_IDX['real']) for p in real_files]
        if len(self.samples) == 0:
            raise RuntimeError("No images found in the provided directories.")
        self.samples.sort(key=lambda x: x[0])  # stable ordering

        if self.use_opencv:
            # lazy import; only if requested
            import cv2  # noqa: F401

    def __len__(self):
        return len(self.samples)

    # --- robust open helpers ---
    def _open_rgb_pil(self, path: str) -> Image.Image:
        img = Image.open(path)     # ImageFile.LOAD_TRUNCATED_IMAGES is set globally
        img.load()                 # force decode to surface errors here
        return img.convert('RGB')

    def _open_rgb_cv2(self, path: str) -> Image.Image:
        import cv2
        arr = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if arr is None:
            raise OSError("cv2 failed to read")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr)

    def _open_rgb(self, path: str) -> Image.Image:
        if self.use_opencv:
            return self._open_rgb_cv2(path)
        return self._open_rgb_pil(path)

    def __getitem__(self, idx: int):
        tries = 0
        cur_idx = idx
        while tries <= self.max_bad_retries:
            path, target = self.samples[cur_idx]
            try:
                img = self._open_rgb(path)
                if self.transform:
                    img = self.transform(img)
                return img, target
            except Exception:
                # pick another index to keep the epoch moving
                cur_idx = random.randrange(0, len(self.samples))
                tries += 1

        # last resort: return a dummy image to avoid crashing the worker
        dummy_img = Image.new('RGB', (self.fallback_size, self.fallback_size), (0, 0, 0))
        if self.transform:
            dummy_img = self.transform(dummy_img)
        else:
            # if no transform, return a zero tensor
            dummy_img = torch.zeros(3, self.fallback_size, self.fallback_size)
        return dummy_img, 0  # arbitrary label; won't matter rarely



class _SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, i):
        base_ds = self.subset.dataset
        path, target = base_ds.samples[self.subset.indices[i]]
        # Use the same robust opener if available (so OpenCV also applies to val)
        if hasattr(base_ds, "_open_rgb"):
            im = base_ds._open_rgb(path)
        else:
            im = Image.open(path).convert('RGB')
        im = self.transform(im)
        return im, target


# -----------------------------
# Original model name plumbing
# -----------------------------
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
my_models = sorted(name for name in model_zoo.__dict__
                   if name.islower() and not name.startswith("__")
                   and callable(model_zoo.__dict__[name]))

model_names.extend(my_models)

parser = argparse.ArgumentParser(description='PyTorch Universal Deepfake Training')

# NEW: CSV-driven input and val split
parser.add_argument('--csv', type=str, default='',
                    help='CSV with columns "real" and/or "fake" listing directory paths; header optional.')
parser.add_argument('--val-split', type=float, default=0.10,
                    help='Fraction of images reserved for validation (e.g., 0.05 for 5%).')
parser.add_argument('--extensions', nargs='+', default=IMG_EXTS_DEFAULT,
                    help='Allowed image extensions (case-insensitive).')

# Keep original -data for backward compatibility (if --csv not supplied)
parser.add_argument('-data', metavar='DIR',
                    help='path to dataset (expects train/ and val/ subfolders)')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--ratio', type=float, default=0.5)
parser.add_argument('--lfu', default=False, action="store_true")
parser.add_argument('--use_se', default=False, action="store_true")
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_steps', default=[30, 60, 80], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--coslr', action='store_true')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training')
parser.add_argument('--root_log', type=str, default='log_big')
parser.add_argument('--root_model', type=str, default='checkpoint_big')
parser.add_argument('--store_name', type=str, default="")
parser.add_argument('--fast-aug', action='store_true',
                    help='Use lighter, faster augmentations')

best_acc1 = 0
best_avg = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, best_avg
    args.gpu = gpu


    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.arch in my_models:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('ffc_'):
            model = model_zoo.__dict__[args.arch](ratio=args.ratio, lfu=args.lfu, use_se=args.use_se)
        else:
            model = model_zoo.__dict__[args.arch]()
        if args.pretrained:
            print("=> WARNING: Missing pretrained model.")
    elif args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    print(model)

    if args.store_name == '':
        args.store_name = args.arch

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True  # <—
            )
            print("Use GPU: {} for training".format(args.gpu))
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True  # <—
            )
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None and hasattr(best_acc1, 'to'):
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # -----------------------------
    # DATA LOADING (CSV mode or fallback to original)
    # -----------------------------
    use_csv_mode = (args.csv is not None and args.csv.strip() != '')

    if use_csv_mode:
        # Build transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(299, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            normalize,
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(333),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize,
        ])

        real_dirs, fake_dirs = _read_dir_csv(args.csv)
        print(f"=> CSV mode: {len(real_dirs)} real dir(s), {len(fake_dirs)} fake dir(s).")

        full_dataset = FolderListImageDataset(
            fake_dirs=fake_dirs,
            real_dirs=real_dirs,
            exts=args.extensions,
            transform=train_transform,
            use_opencv=True,          # <-- enable OpenCV decode
        )

        # Split into train/val
        val_frac = max(0.0, min(0.5, float(args.val_split)))
        n_total = len(full_dataset)
        n_val = int(round(val_frac * n_total))
        n_train = n_total - n_val
        print(f"=> Total images: {n_total} | Train: {n_train} | Val: {n_val} (val-split={val_frac:.2f})")

        g = torch.Generator()
        if args.seed is not None:
            g.manual_seed(args.seed)
        train_dataset, val_subset = random_split(full_dataset, [n_train, n_val], generator=g)
        val_dataset = _SubsetWithTransform(val_subset, eval_transform)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,                 # <—
            persistent_workers=True if args.workers > 0 else False,
            prefetch_factor=4 if args.workers > 0 else None,

        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

        # In CSV mode, test_loader is just the validation split
        test_loader = val_loader

    else:
        # ---------- ORIGINAL BEHAVIOR (expects <data>/train and <data>/val) ----------
        traindir = os.path.join(args.data, 'train')
        testdir = os.path.join(args.data, 'val')

        # Training dataset with strong augmentation
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(299, scale=(0.7, 1.0)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.4, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                normalize,
            ])
        )

        test_dataset = datasets.ImageFolder(
            testdir,
            transforms.Compose([
                transforms.Resize(333),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                normalize,
            ])
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        # Your original 80/20 split from train -> (train/val)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # -----------------------------
    # Training / Evaluation
    # -----------------------------
    check_rootfolders(args)

    # Only rank-0 writes args.txt and creates a real SummaryWriter
    tf_writer = NullWriter()
    if (not args.distributed) or (args.rank in (-1, 0)):
        with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
            f.write(str(args))
        tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    # Optional: let other ranks wait until dirs exist
    if args.distributed:
        dist.barrier()

    if args.evaluate:
        validate(test_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and 'train_sampler' in locals() and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, tf_writer)

        # evaluate on validation set
        acc1, avg = validate(val_loader, model, criterion, args, tf_writer, epoch=epoch)

        try:
            tf_writer.close()
        except Exception:
            pass

        # remember best acc@1 and save checkpoint
        is_best = avg > best_avg
        best_acc1 = max(acc1, best_acc1)
        # NOTE: bugfix — previously best_avg = max(avg, best_acc1) mixed metrics; keep avg
        # but to preserve original logic while fixing: use best_avg vs avg only
        globals()['best_avg'] = max(best_avg, avg)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_avg': best_avg,
                'optimizer': optimizer.state_dict(),
            }, is_best, args)

    print(args.arch, 'best accuracy:', best_acc1)


def train(train_loader, model, criterion, optimizer, epoch, args, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    average = AverageMeter('Acc@avg', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, top5, average,
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        step = epoch * len(train_loader) + i
        adjust_learning_rate(optimizer, epoch, step, len(train_loader), args)

        # compute output
        output = model(input)
        if isinstance(output, (tuple, list)):
            output = output[0]            # logits tensor used for CE
        elif isinstance(output, dict):
            output = output['logits']
        loss = criterion(output, target)

        # measure accuracy and record loss (your code computes avg as overall %)
        acc1, acc5, avg = accuracy(output, target, topk=(1, 1))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        average.update(avg[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
            tf_writer.add_scalar('loss/train', losses.avg, epoch * len(train_loader) + i)
            tf_writer.add_scalar('acc/train_top1', top1.avg, epoch * len(train_loader) + i)
            tf_writer.add_scalar('acc/train_top5', top5.avg, epoch * len(train_loader) + i)
            tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch * len(train_loader) + i)


def validate(val_loader, model, criterion, args, tf_writer=None, epoch=0):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    average = AverageMeter('Acc@avg', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, average,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5, avg = accuracy(output, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            average.update(avg[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Acc@avg {average.avg:.3f}'
              .format(top1=top1, top5=top5, average=average))

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
        tf_writer.add_scalar('acc/test_avg', average.avg, epoch)

    return top1.avg, average.avg


def save_checkpoint(state, is_best, args):
    filename = '{}/{}/fixFULL-ckpt.pth.tar'.format(args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    """Warmup + cosine/step as per your original logic."""
    if epoch < args.warmup_epochs:
        lr = args.lr * float(step) / max(1, (args.warmup_epochs * len_epoch))
    elif args.coslr:
        nmax = len_epoch * args.epochs
        lr = args.lr * 0.5 * (np.cos(step / max(1, nmax) * np.pi) + 1)
    else:
        decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
        lr = args.lr * decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions; also returns overall % as 'avg'."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        avg = correct[:].reshape(-1).float().sum(0, keepdim=True)
        avg = avg.mul_(100.0 / batch_size)
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        res.append(avg)
        return res


def check_rootfolders(args):
    """Create/backup log & model folders. Only rank-0 does this. Collision-safe."""
    # Only one process should touch the filesystem layout
    if getattr(args, "distributed", False) and getattr(args, "rank", -1) not in (-1, 0):
        return

    folders = [
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name),
    ]
    ts = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

    for folder in folders:
        if os.path.isdir(folder):
            base = f"{folder}_{ts}_backup"
            dst = base
            i = 1
            # ensure uniqueness even if multiple runs hit the same second
            while os.path.exists(dst):
                i += 1
                dst = f"{base}.{i}"
            print(f"[folders] moving existing '{folder}' -> '{dst}'")
            shutil.move(folder, dst)

        # dirs_exist_ok from py3.8 isn't available in os.makedirs; emulate:
        if not os.path.isdir(folder):
            print('creating folder ' + folder)
            os.makedirs(folder, exist_ok=True)



if __name__ == '__main__':
    main()
