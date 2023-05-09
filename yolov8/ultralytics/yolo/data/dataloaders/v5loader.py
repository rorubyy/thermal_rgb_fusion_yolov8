# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Dataloaders and dataset utils
"""

import contextlib
import glob
import hashlib
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import psutil
import torch
import torchvision
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from ultralytics.yolo.utils import (DATASETS_DIR, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, is_colab, is_dir_writeable,
                                    is_kaggle)
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.ops import clean_str, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn
from ultralytics.yolo.utils.torch_utils import torch_distributed_zero_first

from .v5augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                              letterbox, mixup, random_perspective, random_perspective_rgb_ir)

# Parameters
HELP_URL = 'See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90}.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info['exif'] = exif.tobytes()
    return image


def seed_worker(worker_id):
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(path_rgb,
                      path_ir,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      close_mosaic=False,
                      min_items=0,
                      prefix='',
                      shuffle=False,
                      seed=0):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
            path_rgb,
            path_ir,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            min_items=min_items,
            prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights or close_mosaic else InfiniteDataLoader  # DataLoader allows attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=LoadImagesAndLabels.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that reuses workers for same syntax as vanilla DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of batch_sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that infinitely repeats."""
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """Sampler that repeats forever

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Sampler that repeats dataset samples infinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Infinite loop iterating over a given sampler."""
        while True:
            yield from iter(self.sampler)


class LoadScreenshots:
    # YOLOv5 screenshot dataloader, i.e. `python detect.py --source "screen 0 100 100 512 256"`
    def __init__(self, source, img_size=640, stride=32, auto=True, transforms=None):
        """source = [screen_number left top width height] (pixels)."""
        check_requirements('mss')
        import mss

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.img_size = img_size
        self.stride = stride
        self.transforms = transforms
        self.auto = auto
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor['top'] if top is None else (monitor['top'] + top)
        self.left = monitor['left'] if left is None else (monitor['left'] + left)
        self.width = width or monitor['width']
        self.height = height or monitor['height']
        self.monitor = {'left': self.left, 'top': self.top, 'width': self.width, 'height': self.height}

    def __iter__(self):
        """Iterates over objects with the same structure as the monitor attribute."""
        return self

    def __next__(self):
        """mss screen capture: get raw pixels from the screen as np array."""
        im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
        s = f'screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: '

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        self.frame += 1
        return str(self.screen), im, im0, None, s  # screen, img, original img, im0s, s


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """Initialize instance variables and check for valid input."""
        if isinstance(path, str) and Path(path).suffix == '.txt':  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        """Returns an iterator object for iterating over images or videos found in a directory."""
        self.count = 0
        return self

    def __next__(self):
        """Iterator's next item, performs transformation on image and returns path, transformed image, original image, capture and size."""
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        """Create a new video capture object."""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        """Rotate a cv2 video manually."""
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        """Returns the number of files in the class instance."""
        return self.nf  # number of files


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='file.streams', img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """Initialize YOLO detector with optional transforms and check input shapes."""
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/Zgi9g1ksQHc'
                check_requirements(('pafy', 'youtube_dl==2020.12.2'))
                import pafy
                s = pafy.new(s).getbest(preftype='mp4').url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0:
                assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
                assert not is_kaggle(), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f'{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)')
            self.threads[i].start()
        LOGGER.info('')  # newline

        # Check for common shapes
        s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        """Read stream `i` frames in daemon thread."""
        n, f = 0, self.frames[i]  # frame number, frame array
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()  # .read() = .grab() followed by .retrieve()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        """Iterator that returns the class instance."""
        self.count = -1
        return self

    def __next__(self):
        """Return a tuple containing transformed and resized image data."""
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, ''

    def __len__(self):
        """Returns the number of sources as the length of the object."""
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):
    """YOLOv5 train_loader/val_loader, loads images and labels for training and validation."""
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 path_rgb,
                 path_ir,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path_rgb = path_rgb
        self.path_ir = path_ir
        self.albumentations = Albumentations(size=img_size) if augment else None

        try:
            f_rgb = []  # image file
            f_ir = []

            # -----------------------------  rgb   -----------------------------
            for p_rgb in path_rgb if isinstance(path_rgb, list) else[path_rgb]:
                p_rgb = Path(p_rgb)  # os-agnostic
                if p_rgb.is_dir():  # dir
                    f_rgb += glob.glob(str(p_rgb / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p_rgb.is_file():  # file
                    with open(p_rgb, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p_rgb.parent) + os.sep
                        f_rgb += [x.replace('./', parent) if x.startswith('./')
                                  else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{path_rgb} does not exist')

            # -----------------------------  ir   -----------------------------
            for p_ir in path_ir if isinstance(path_ir, list) else [path_ir]:
                p_ir = Path(p_ir)  # os-agnostic
                if p_ir.is_dir():  # dir
                    f_ir += glob.glob(str(p_ir / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p_ir.is_file():  # file
                    with open(p_ir, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p_ir.parent) + os.sep
                        f_ir += [x.replace('./', parent) if x.startswith('./') else x for x in
                                 t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p_ir} does not exist')

            self.img_files_rgb = sorted([x.replace('/', os.sep)
                                        for x in f_rgb if x.split('.')[-1].lower() in IMG_FORMATS])
            self.img_files_ir = sorted([x.replace('/', os.sep)
                                       for x in f_ir if x.split('.')[-1].lower() in IMG_FORMATS])

            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert (self.img_files_rgb, self.img_files_ir), (f'{prefix}No images found', f'{prefix}No images found')
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path_rgb,path_ir}: {e}\nSee {HELP_URL}')

        # Check RGB cache
        self.label_files_rgb = img2label_paths(self.img_files_rgb)  # labels
        cache_rgb_path = (p_rgb if p_rgb.is_file() else Path(
            self.label_files_rgb[0]).parent).with_suffix('.cache')  # cached labels
        # if cache_rgb_path.is_file():
        #     cache_rgb, exists_rgb = torch.load(cache_rgb_path), True  # load
        #     if cache_rgb['hash'] != get_hash(self.label_files_rgb + self.img_files_rgb) or 'version' not in cache_rgb:  # changed
        #         cache_rgb, exists_rgb = self.cache_labels(self.img_files_rgb, self.label_files_rgb,
        #                                                   cache_rgb_path, prefix), False  # re-cache
        # else:
        cache_rgb, exists_rgb = self.cache_labels(self.img_files_rgb, self.label_files_rgb,
                                                  cache_rgb_path, prefix), False  # cache

        # Check IR cache
        self.label_files_ir = img2label_paths(self.img_files_ir)  # labels
        # print(self.label_files)
        cache_ir_path = (p_ir if p_ir.is_file() else Path(
            self.label_files_ir[0]).parent).with_suffix('.cache')  # cached labels
        # if cache_ir_path.is_file():
        #     cache_ir, exists_ir = torch.load(cache_ir_path), True  # load
        #     if cache_ir['hash'] != get_hash(self.label_files_ir + self.img_files_ir) or 'version' not in cache_ir:  # changed
        #         cache_ir, exists_ir = self.cache_labels(self.img_files_ir, self.label_files_ir,
        #                                                 cache_ir_path, prefix), False  # re-cache
        # else:
        cache_ir, exists_ir = self.cache_labels(self.img_files_ir, self.label_files_ir,
                                                    cache_ir_path, prefix), False  # cache
        # Display cache
        nf_rgb, nm_rgb, ne_rgb, nc_rgb, n_rgb = cache_rgb.pop('results')  # found, missing, empty, corrupted, total
        nf_ir, nm_ir, ne_ir, nc_ir, n_ir = cache_ir.pop('results')  # found, missing, empty, corrupted, total
        if exists_rgb:
            d = f"Scanning RGB '{cache_rgb_path}' images and labels... {nf_rgb} found, {nm_rgb} missing, {ne_rgb} empty, {nc_rgb} corrupted"
            tqdm(None, desc=prefix + d, total=n_rgb, initial=n_rgb)  # display cache results
        if exists_ir:
            d = f"Scanning IR '{cache_rgb_path}' images and labels... {nf_ir} found, {nm_ir} missing, {ne_ir} empty, {nc_ir} corrupted"
            tqdm(None, desc=prefix + d, total=n_ir, initial=n_ir)  # display cache results

        assert nf_rgb > 0 or not augment, f'{prefix}No labels in {cache_rgb_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        # Read RGB cache
        cache_rgb.pop('hash')  # remove hash
        cache_rgb.pop('version')  # remove version
        cache_rgb.pop('msgs')
        labels_rgb, shapes_rgb, self.segments_rgb = zip(*cache_rgb.values())
        self.labels_rgb = list(labels_rgb)
        self.shapes_rgb = np.array(shapes_rgb, dtype=np.float64)
        self.img_files_rgb = list(cache_rgb.keys())  # update
        self.label_files_rgb = img2label_paths(cache_rgb.keys())  # update
        if single_cls:
            for x in self.labels_rgb:
                x[:, 0] = 0

        n_rgb = len(shapes_rgb)  # number of images
        bi_rgb = np.floor(np.arange(n_rgb) / batch_size).astype(np.int)  # batch index
        nb_rgb = bi_rgb[-1] + 1  # number of batches
        self.batch_rgb = bi_rgb  # batch index of image
        self.n_rgb = n_rgb
        self.indices_rgb = range(n_rgb)

        # Read IR cache
        cache_ir.pop('hash')  # remove hash
        cache_ir.pop('version')  # remove version.
        cache_ir.pop('msgs')
        labels_ir, shapes_ir, self.segments_ir = zip(*cache_ir.values())
        self.labels_ir = list(labels_ir)
        self.shapes_ir = np.array(shapes_ir, dtype=np.float64)
        self.img_files_ir = list(cache_ir.keys())  # update
        self.label_files_ir = img2label_paths(cache_ir.keys())  # update
        if single_cls:
            for x in self.labels_ir:
                x[:, 0] = 0

        n_ir = len(shapes_ir)  # number of images
        bi_ir = np.floor(np.arange(n_ir) / batch_size).astype(np.int)  # batch index
        nb_ir = bi_ir[-1] + 1  # number of batches
        self.batch_ir = bi_ir  # batch index of image
        self.n_ir = n_ir
        self.indices_ir = range(n_ir)

        # Rectangular Training
        if self.rect:

            # RGB
            # Sort by aspect ratio
            s_rgb = self.shapes_rgb  # wh
            ar_rgb = s_rgb[:, 1] / s_rgb[:, 0]  # aspect ratio
            irect_rgb = ar_rgb.argsort()
            self.img_files_rgb = [self.img_files_rgb[i] for i in irect_rgb]
            self.label_files_rgb = [self.label_files_rgb[i] for i in irect_rgb]
            self.labels_rgb = [self.labels_rgb[i] for i in irect_rgb]
            self.shapes_rgb = s_rgb[irect_rgb]  # wh
            ar_rgb = ar_rgb[irect_rgb]

            # Set training image shapes
            shapes_rgb = [[1, 1]] * nb_rgb
            for i in range(nb_rgb):
                ari_rgb = ar_rgb[bi_rgb == i]
                mini, maxi = ari_rgb.min(), ari_rgb.max()
                if maxi < 1:
                    shapes_rgb[i] = [maxi, 1]
                elif mini > 1:
                    shapes_rgb[i] = [1, 1 / mini]

            self.batch_shapes_rgb = np.ceil(np.array(shapes_rgb) * img_size / stride + pad).astype(int) * stride

            # IR
            # Sort by aspect ratio
            s_ir = self.shapes_ir  # wh
            ar_ir = s_ir[:, 1] / s_ir[:, 0]  # aspect ratio
            irect_ir = ar_ir.argsort()
            self.img_files_ir = [self.img_files_ir[i] for i in irect_ir]
            self.label_files_ir = [self.label_files_ir[i] for i in irect_ir]
            self.labels_ir = [self.labels_ir[i] for i in irect_ir]
            self.shapes_ir = s_ir[irect_ir]  # wh
            ar_ir = ar_ir[irect_ir]

            # Set training image shapes
            shapes_ir = [[1, 1]] * nb_ir
            for i in range(nb_ir):
                ari_ir = ar_ir[bi_ir == i]
                mini, maxi = ari_ir.min(), ari_ir.max()
                if maxi < 1:
                    shapes_ir[i] = [maxi, 1]
                elif mini > 1:
                    shapes_ir[i] = [1, 1 / mini]

            self.batch_shapes_rgb = np.ceil(np.array(shapes_ir) * img_size / stride + pad).astype(int) * stride

        self.img_rgb = [None] * n_rgb
        self.img_ir = [None] * n_ir

        self.labels = self.labels_rgb
        self.shapes = self.shapes_rgb
        self.indices = self.indices_rgb
        # self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]

    def check_cache_ram(self, safety_margin=0.1, prefix=''):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.n, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.n / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(f'{prefix}{mem_required / gb:.1f}GB RAM required, '
                        f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def cache_labels(self, img_file, label_file, path=Path('./labels.cache'), prefix=''):
        """Cache labels and save as numpy file for next time."""
        # Cache dataset labels, check images and read shapes
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{prefix}Scanning {path.parent / path.stem}...'
        total = len(self.img_files_rgb)

        with ThreadPool(NUM_THREADS) as pool:
            pbar = tqdm(zip(img_file, label_file), desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            for i, (im_file, lb_file) in enumerate(pbar):
                try:
                    # verify images
                    im = Image.open(im_file)
                    im.verify()  # PIL verify
                    shape = exif_size(im)  # image size
                    segments = []  # instance segments
                    assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                    assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'

                    # verify labels
                    if os.path.isfile(lb_file):
                        nf += 1  # label found
                        with open(lb_file, 'r') as f:
                            l = [x.split() for x in f.read().strip().splitlines()]
                            if any([len(x) > 8 for x in l]):  # is segment
                                classes = np.array([x[0] for x in l], dtype=np.float32)
                                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2)
                                            for x in l]  # (cls, xy1...)
                                l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                            l = np.array(l, dtype=np.float32)
                        if len(l):
                            assert l.shape[1] == 5, 'labels require 5 columns each'
                            assert (l >= 0).all(), 'negative labels'
                            assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                            assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                        else:
                            ne += 1  # label empty
                            l = np.zeros((0, 5), dtype=np.float32)
                    else:
                        nm += 1  # label missing
                        l = np.zeros((0, 5), dtype=np.float32)
                    x[im_file] = [l, shape, segments]
                except Exception as e:
                    nc += 1
                    print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')

        x['hash'] = get_hash(label_file + img_file)
        x['results'] = nf, nm, ne, nc, i + 1
        x['msgs'] = msgs  # warnings
        x['version'] = 0.1  # cache version

        if is_dir_writeable(path.parent):
            np.save(str(path), x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        else:
            LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable')  # not writeable
        return x

    def __len__(self):
        """Returns the length of 'im_files' attribute."""
        return len(self.img_files_rgb)

    def __getitem__(self, index):
        """Get a sample and its corresponding label, filename and shape from the dataset."""
        index_rgb = self.indices_rgb[index]  # linear, shuffled, or image_weights
        index_ir = self.indices_ir[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img_rgb, labels_rgb, img_ir, labels_ir = self.load_mosaic(index_rgb, index_ir)
            shapes = None

        else:
            # Load image
            img_rgb, img_ir, (h0, w0), (h, w) = self. load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img_rgb, ratio, pad = letterbox(img_rgb, shape, auto=False, scaleup=self.augment)
            img_ir, ratio, pad = letterbox(img_ir, shape, auto=False, scaleup=self.augment)

            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels_rgb[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            labels_rgb = labels
            labels_ir = labels

            if self.augment:
                augment_hsv(img_rgb, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
                augment_hsv(img_ir, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        nl = len(labels_rgb)  # number of labels
        if nl:
            labels_rgb[:, 1:5] = xyxy2xywhn(labels_rgb[:, 1:5], w=img_rgb.shape[1],
                                            h=img_rgb.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img_rgb, labels_rgb = self.albumentations(img_rgb, labels_rgb)
            img_ir, labels_ir = self.albumentations(img_ir, labels_ir)

            nl = len(labels_rgb)  # update after albumentations

            # HSV color-space
            augment_hsv(img_rgb, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            augment_hsv(img_ir, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img_rgb = np.flipud(img_rgb)
                img_ir = np.flipud(img_ir)
                if nl:
                    labels_rgb[:, 2] = 1 - labels_rgb[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img_rgb = np.fliplr(img_rgb)
                img_ir = np.fliplr(img_ir)
                if nl:
                    labels_rgb[:, 1] = 1 - labels_rgb[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels_rgb)

        # Convert
        img_rgb = img_rgb[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_rgb = np.ascontiguousarray(img_rgb)
        img_ir = img_ir[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_ir = np.ascontiguousarray(img_ir)

        img_all = np.concatenate((img_rgb, img_ir), axis=0)

        return torch.from_numpy(img_all), labels_out, self.img_files_rgb[index], shapes

    def load_image(self, i):
        img_rgb = self.img_rgb[i]
        img_ir = self.img_ir[i]

        if (img_rgb is None) and (img_ir is None):  # not cached
            path_rgb = self.img_files_rgb[i]
            path_ir = self.img_files_ir[i]
            img_rgb = cv2.imread(path_rgb)  # BGR
            img_ir = cv2.imread(path_ir)  # BGR

            assert img_rgb is not None, 'Image RGB Not Found ' + path_rgb
            assert img_ir is not None, 'Image IR Not Found ' + path_ir

            h0, w0 = img_rgb.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio

            if r != 1:  # if sizes are not equal
                img_rgb = cv2.resize(img_rgb, (int(w0 * r), int(h0 * r)),
                                     interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
                img_ir = cv2.resize(img_ir, (int(w0 * r), int(h0 * r)),
                                    interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            return img_rgb, img_ir, (h0, w0), img_rgb.shape[:2]  # img, hw_original, hw_resized
        else:
            return self.imgs_rgb[i], self.imgs_ir[i], self.img_hw0_rgb[i], self.img_hw_rgb[i]  # img, hw_original, hw_resized        """Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)."""

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def load_mosaic(self, index1, index2):
        """YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic."""
        index_rgb = index1
        index_ir = index2

        labels4_rgb, segments4_rgb = [], []
        labels4_ir, segments4_ir = [], []

        s = self.img_size

        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

        assert index_rgb == index_ir, 'INDEX RGB != INDEX IR'

        indices = [index_rgb] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)

        for i, index in enumerate(indices):
            # Load image
            img_rgb, img_ir, _, (h, w) = self.load_image(index)

            # Place img in img4
            if i == 0:  # top left
                img4_rgb = np.full((s * 2, s * 2, img_rgb.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                img4_ir = np.full((s * 2, s * 2, img_ir.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4_rgb[y1a:y2a, x1a:x2a] = img_rgb[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            img4_ir[y1a:y2a, x1a:x2a] = img_ir[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels_rgb, segments_rgb = self.labels_rgb[index].copy(), self.segments_rgb[index].copy()
            labels_ir, segments_ir = self.labels_ir[index].copy(), self.segments_ir[index].copy()
            if labels_rgb.size:
                # normalized xywh to pixel xyxy format
                labels_rgb[:, 1:] = xywhn2xyxy(labels_rgb[:, 1:], w, h, padw, padh)
                # normalized xywh to pixel xyxy format
                labels_ir[:, 1:] = xywhn2xyxy(labels_ir[:, 1:], w, h, padw, padh)
                segments_rgb = [xyn2xy(x, w, h, padw, padh) for x in segments_rgb]
                segments_ir = [xyn2xy(x, w, h, padw, padh) for x in segments_ir]

            labels4_rgb.append(labels_rgb)
            segments4_rgb.extend(segments_rgb)
            labels4_ir.append(labels_ir)
            segments4_ir.extend(segments_ir)

        # Concat/clip labels
        labels4_rgb = np.concatenate(labels4_rgb, 0)
        labels4_ir = np.concatenate(labels4_ir, 0)
        for x in (labels4_rgb[:, 1:], *segments4_rgb):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        for x in (labels4_ir[:, 1:], *segments4_ir):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4_rgb, img4_ir, labels4_rgb, labels4_ir = random_perspective_rgb_ir(img4_rgb, img4_ir, labels4_rgb, labels4_ir,
                                                                               segments4_rgb, segments4_ir,
                                                                               degrees=self.hyp['degrees'],
                                                                               translate=self.hyp['translate'],
                                                                               scale=self.hyp['scale'],
                                                                               shear=self.hyp['shear'],
                                                                               perspective=self.hyp['perspective'],
                                                                               border=self.mosaic_border)  # border to remove
        labels4_ir = labels4_rgb

        return img4_rgb, labels4_rgb, img4_ir, labels4_ir

    def load_mosaic9(self, index):
        """YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic."""
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # Place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9, labels9, segments9 = copy_paste(img9, labels9, segments9, p=self.hyp['copy_paste'])
        img9, labels9 = random_perspective(img9,
                                           labels9,
                                           segments9,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img9, labels9

    @staticmethod
    def collate_fn(batch):
        """YOLOv8 collate function, outputs dict."""
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        batch_idx, cls, bboxes = torch.cat(label, 0).split((1, 1, 4), dim=1)
        return {
            'ori_shape': tuple((x[0] if x else None) for x in shapes),
            'ratio_pad': tuple((x[1] if x else None) for x in shapes),
            'im_file': path,
            'img': torch.stack(im, 0),
            'cls': cls,
            'bboxes': bboxes,
            'batch_idx': batch_idx.view(-1)}

    @staticmethod
    def collate_fn_old(batch):
        """YOLOv5 original collate function."""
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes


# Ancillary functions --------------------------------------------------------------------------------------------------
def flatten_recursive(path=DATASETS_DIR / 'coco128'):
    """Flatten a recursive directory by bringing all files to top level."""
    new_path = Path(f'{str(path)}_flat')
    if os.path.exists(new_path):
        shutil.rmtree(new_path)  # delete output folder
    os.makedirs(new_path)  # make new output folder
    for file in tqdm(glob.glob(f'{str(Path(path))}/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path=DATASETS_DIR / 'coco128'):  # from utils.dataloaders import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classification') if (path / 'classification').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # Image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # Labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # B[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path=DATASETS_DIR / 'coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write(f'./{img.relative_to(path.parent).as_posix()}' + '\n')  # add image to txt file


def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLOv5 Classification Dataset.
    Arguments
        root:  Dataset path
        transform:  torchvision transforms, used by default
        album_transform: Albumentations transforms, used if installed
    """

    def __init__(self, root, augment, imgsz, cache=False):
        """Initialize YOLO dataset with root, augmentation, image size, and cache parameters."""
        super().__init__(root=root)
        self.torch_transforms = classify_transforms(imgsz)
        self.album_transforms = classify_albumentations(augment, imgsz) if augment else None
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im

    def __getitem__(self, i):
        """Retrieves data items of 'dataset' via indices & creates InfiniteDataLoader."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
        else:
            sample = self.torch_transforms(im)
        return sample, j


def create_classification_dataloader(path,
                                     imgsz=224,
                                     batch_size=16,
                                     augment=True,
                                     cache=False,
                                     rank=-1,
                                     workers=8,
                                     shuffle=True):
    """Returns Dataloader object to be used with YOLOv5 Classifier."""
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = ClassificationDataset(root=path, imgsz=imgsz, augment=augment, cache=cache)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              worker_init_fn=seed_worker,
                              generator=generator)  # or DataLoader(persistent_workers=True)
