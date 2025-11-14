import torch
from torchvision import datasets, transforms
import torchray.benchmark.datasets as dats
import torchray.benchmark.models as pmodels
import os
from cub_tools.transforms import makeDefaultTransforms

def get_dataset(dataset_name, train=False):
    if dataset_name == 'imagenet':
        datadir = os.environ["IMAGENETDIR"]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        cur_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])
        if train:
            datadir = os.path.join(datadir, "train")
        else:
            datadir = os.path.join(datadir, "val")

        cur_data = datasets.ImageFolder(
                datadir,
                cur_transform
                )

    elif dataset_name == 'cifar10':
        datadir = './data'
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

        cur_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        if train:
            cur_data = datasets.CIFAR10(datadir, train=True, download=True, transform=cur_transform)
        else:
            cur_data = datasets.CIFAR10(datadir, train=False, download=True, transform=cur_transform)


    elif dataset_name == "coco":
        transform = pmodels.get_transform("coco")
        cur_data = dats.get_dataset("coco", "val2014", transform=transform)
    elif dataset_name == "voc":
        transform = pmodels.get_transform("voc")
        cur_data = dats.get_dataset("voc_2007", subset='test', transform=transform)

    elif dataset_name == "cub":
        datadir = os.environ.get("CUBDIR")
        if datadir is None:
            raise RuntimeError("CUBDIR env var not set (path to CUB_200_2011)")
        data_transforms = makeDefaultTransforms()
        if train:
            cur_data = datasets.ImageFolder(os.path.join(datadir, 'images', 'train'), data_transforms['train'])
        else:
            cur_data = datasets.ImageFolder(os.path.join(datadir, 'images', 'test'), data_transforms['test'])
    else:
        raise RuntimeError("dataset not supported")
    return cur_data

def select_data(dataset, idx, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imgs, data_category = dataset[idx]
    imgs = imgs.unsqueeze(0).to(device)
    with torch.no_grad():
        model_category = model(imgs).max(1)[1].item()

    return imgs, data_category, model_category



def select_category(dataset, category, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    targets = torch.tensor(dataset.targets)
    idxs = torch.argwhere(targets == category).squeeze().cpu().numpy().tolist()
    inputs = []
    chosen_idxs = []
    for i in idxs:
        img, target = dataset[i]
        if target == category:
            img_batch = img.unsqueeze(0).to(device)
            with torch.no_grad():
                model_target = model(img_batch).max(1)[1]
            if model_target.item() == target:
                inputs.append(img_batch)
                chosen_idxs.append(i)
            else:
                print(i)

    subset = torch.utils.data.Subset(dataset, list(chosen_idxs))
    imgs = torch.cat(inputs)
    return imgs, subset, chosen_idxs


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
    'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]

_COCO_CLASS_TO_INDEX = {c: i for i, c in enumerate([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
    ])}


def FromVOCToClasses(label):
    objs = label['annotation']['object']
    if not isinstance(objs, list):
        objs = [objs]
    classes = [VOC_CLASSES.index(obj['name']) for obj in objs]
    return classes

def FromCocoToClasses(label):
    classes = [_COCO_CLASS_TO_INDEX[item["category_id"]] for item in label]
    return classes


