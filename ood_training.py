import argparse
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import config_training_setup
from src.imageaugmentations import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from src.model_utils import load_network
from torch.utils.data import DataLoader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.projects.panoptic_deeplab import (
    add_panoptic_deeplab_config
)
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
import config as meta_ood_config


def cross_entropy(logits, targets):
    """
    cross entropy loss with one/all hot encoded targets -> logits.size()=targets.size()
    :param logits: torch tensor with logits obtained from network forward pass
    :param targets: torch tensor one/all hot encoded
    :return: computed loss
    """
    neg_log_like = - 1.0 * F.log_softmax(logits, 1)
    L = torch.mul(targets.float(), neg_log_like)
    L = L.mean()
    return L

def deep_lab_loss(logits, targets, criterion):
    #pixel_losses = criterion(logits, targets).contiguous().view(-1)

    neg_log_like = - 1.0 * F.log_softmax(logits, 1)
    L = torch.mul(targets.float(), neg_log_like).sum(dim=1).contiguous().view(-1)
    #L = L.mean()

    top_k_pixels = int(0.2 * L.numel())
    pixel_losses, _ = torch.topk(L, top_k_pixels)
    return pixel_losses.mean()


def encode_target(target, pareto_alpha, num_classes, ignore_train_ind, ood_ind=254):
    """
    encode target tensor with all hot encoding for OoD samples
    :param target: torch tensor
    :param pareto_alpha: OoD loss weight
    :param num_classes: number of classes in original task
    :param ignore_train_ind: void class in original task
    :param ood_ind: class label corresponding to OoD class
    :return: one/all hot encoded torch tensor
    """
    npy = target.numpy()
    npz = npy.copy()
    npy[np.isin(npy, ood_ind)] = num_classes
    npy[np.isin(npy, ignore_train_ind)] = num_classes + 1
    enc = np.eye(num_classes + 2)[npy][..., :-2]  # one hot encoding with last 2 axis cutoff
    enc[(npy == num_classes)] = np.full(num_classes, pareto_alpha / num_classes)  # set all hot encoded vector
    #enc[(enc == 1)] = 1 - pareto_alpha  # convex combination between in and out distribution samples
    enc[np.isin(npz, ignore_train_ind)] = np.zeros(num_classes)
    enc = torch.from_numpy(enc)
    enc = enc.permute(0, 3, 1, 2).contiguous()
    return enc

def panoptic_deep_lab_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.stack(target)
    return data, target

def training_routine(config):
    """Start OoD Training"""
    print("START OOD TRAINING")
    params = config.params
    roots = config.roots
    dataset = config.dataset(split="train", cs_root=roots.cs_root, coco_root=roots.coco_root, cs_split="train", coco_split="val")
    print("Pareto alpha:", params.pareto_alpha)
    start_epoch = params.training_starting_epoch
    epochs = params.num_training_epochs
    start = time.time()

    # This parameter is needed for panoptic segmentation data loading
    dataset_cfg = None

    # load configuration from cfg files for detectron2
    cfg = get_cfg()
    if roots.model_name == "Detectron_DeepLab":
        add_deeplab_config(cfg)
        cfg.merge_from_file(meta_ood_config.Detectron_DeepLab_Config)
    elif roots.model_name == "Detectron_Panoptic_DeepLab":
        add_panoptic_deeplab_config(cfg)
        cfg.merge_from_file(meta_ood_config.Detectron_PanopticDeepLab_Config)
        dataset_cfg = cfg

    """Initialize model"""
    if start_epoch == 0:
        network = load_network(model_name=roots.model_name, num_classes=dataset.num_classes,
                               ckpt_path=roots.init_ckpt, train=True, cfg=cfg)
    else:
        basename = roots.model_name + "_epoch_" + str(start_epoch) \
                   + "_alpha_" + str(params.pareto_alpha) + ".pth"
        network = load_network(model_name=roots.model_name, num_classes=dataset.num_classes,
                               ckpt_path=os.path.join(roots.weights_dir, basename), train=True, cfg=cfg)

    transform = Compose([RandomHorizontalFlip(), RandomCrop(params.crop_size), ToTensor(),
                         Normalize(dataset.mean, dataset.std)])

    cr_loss = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="none")

    for epoch in range(start_epoch, start_epoch + epochs):
        """Perform one epoch of training"""
        print('\nEpoch {}/{}'.format(epoch + 1, start_epoch + epochs))
        optimizer = optim.Adam(network.parameters(), lr=params.learning_rate)
        trainloader = config.dataset('train', transform, roots.cs_root, roots.coco_root, params.ood_subsampling_factor, cs_split="train", coco_split="val", cfg=dataset_cfg, model="detectron")
        dataloader = DataLoader(trainloader, batch_size=params.batch_size, shuffle=True, collate_fn=panoptic_deep_lab_collate)
        #dataloader = DataLoader(trainloader, batch_size=params.batch_size, shuffle=True)
        i = 0
        loss = None
        for x, target in dataloader:
            optimizer.zero_grad()
            logits, losses = network(x)
            #logits = network(x.cuda())
            logits = logits["sem_seg_results"]

            '''import matplotlib.pyplot as plt
            plt.imshow(x[0]["image"].permute(1, 2, 0).numpy())
            #plt.imshow(x[0].permute(1, 2, 0).numpy())
            plt.show()
            out = torch.squeeze(logits[0]).detach().cpu().numpy().argmax(axis=0)
            plt.imshow(out)
            plt.show()
            plt.imshow(x[0]["sem_seg"])
            plt.show()
            plt.imshow(torch.squeeze(target[0]).numpy())
            plt.show()'''

            y = encode_target(target=target, pareto_alpha=params.pareto_alpha, num_classes=dataset.num_classes,
                              ignore_train_ind=dataset.void_ind, ood_ind=dataset.train_id_out).cuda()



            ''' plt.imshow(np.max(torch.squeeze(y[0]).cpu().numpy(), axis=0))
            plt.show()'''
            #l1 = cr_loss(logits, x[0]["sem_seg"].unsqueeze(dim=0).cuda())
            loss = deep_lab_loss(logits, y, cr_loss)
            #loss = cross_entropy(logits, y)

            loss.backward()
            optimizer.step()
            print('{} Loss: {}'.format(i, loss.item()))
            i += 1

        """Save model state"""
        save_basename = roots.model_name + "_epoch_" + str(epoch + 1) + "_alpha_" + str(params.pareto_alpha) + ".pth"
        print('Saving checkpoint', os.path.join(roots.weights_dir, save_basename))
        torch.save({
            'model': network.state_dict()
        }, os.path.join(roots.weights_dir, save_basename))
        torch.cuda.empty_cache()

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("FINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def main(args):
    """Perform training"""
    config = config_training_setup(args.default_args)
    training_routine(config)


if __name__ == '__main__':
    """Get Arguments and setup config class"""
    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-train", "--TRAINSET", nargs="?", type=str)
    parser.add_argument("-model", "--MODEL", nargs="?", type=str)
    parser.add_argument("-epoch", "--training_starting_epoch", nargs="?", type=int)
    parser.add_argument("-nepochs", "--num_training_epochs", nargs="?", type=int)
    parser.add_argument("-alpha", "--pareto_alpha", nargs="?", type=float)
    parser.add_argument("-lr", "--learning_rate", nargs="?", type=float)
    parser.add_argument("-crop", "--crop_size", nargs="?", type=int)

    # use detectron2 distributed args
    args = default_argument_parser().parse_args()
    #args from current file
    args.default_args = vars(parser.parse_args())
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    #main(vars(parser.parse_args()))
