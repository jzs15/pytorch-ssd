import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config, mobilenetv3_ssd_config_240, mobilenetv3_ssd_config_200, mobilenetv3_ssd_config_160
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from torch.utils.tensorboard import SummaryWriter

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer

import numpy as np

from vision.optim.radam import RAdam

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets', type=str, help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")


parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite, mb3-large-ssd-lite, mb3-small-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--image_size', default=300, type=int, choices=[300, 240, 200, 160],
                    help='Input Image size')
parser.add_argument('--lossfunc', default='l1loss', type=str, choices=['l1loss', 'iou', 'giou', 'diou', 'ciou'],
                    help='Input Image size')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")

def is_parallel(model):
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)

def de_parallel(model):
    return model.module if is_parallel(model) else model

def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases

def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)

def train(loader, net, criterion, optimizer, device, epoch=-1, tb_writer=None):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0

    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

#        print('train box: ', boxes)
        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        print('train loss: ', loss.item())
        skip = False
        if torch.isnan(loss) == True:
            skip = True

        if skip == True:
            print('skip this loss!')
            continue

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

        if epoch == 1:
            tb_writer.add_graph(torch.jit.trace(de_parallel(net), images, strict=False), [])  # graph

    return running_loss / num, running_regression_loss / num, running_classification_loss / num


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

def eval(args, net_state_dict, device, iou_threshold, label_file, targetPath, config):
    class_names = [name.strip() for name in open(label_file).readlines()]

    dataset = OpenImagesDataset(args.datasets, dataset_type="test")
    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)

    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    elif args.net == 'mb3-large-ssd-lite':
        net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True, config=config)
    elif args.net == 'mb3-small-ssd-lite':
        net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True, config=config)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    net.load_state_dict(net_state_dict)
    net = net.to(device)

    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method='hard', device=DEVICE)
    elif args.net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, nms_method='hard', device=DEVICE)
    elif args.net == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method='hard', device=DEVICE)
    elif args.net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net,nms_method='hard', device=DEVICE)
    elif args.net == 'mb2-ssd-lite' or args.net == "mb3-large-ssd-lite" or args.net == "mb3-small-ssd-lite":
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method='hard', device=DEVICE, config=config)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    results = []
    for i in range(len(dataset)):
        image = dataset.get_image(i)
        boxes, labels, probs = predictor.predict(image)
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))
    results = torch.cat(results)

    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = targetPath + f"/det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )

    aps = []
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = targetPath + f"/det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            iou_threshold,
            False
        )
        aps.append(ap)

#    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")

    return class_names, aps

def cal_boxdiff(args, net_state_dict, DEVICE, iou_treshold, label_file, config):
    class_names = [name.strip() for name in open(label_file).readlines()]

    dataset = OpenImagesDataset(args.datasets, dataset_type="test")

    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    elif args.net == 'mb3-large-ssd-lite':
        net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True, config=config)
    elif args.net == 'mb3-small-ssd-lite':
        net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True, config=config)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    net.load_state_dict(net_state_dict)
    net = net.to(DEVICE)

    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method='hard', device=DEVICE, candidate_size=200)
    elif args.net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, nms_method='hard', device=DEVICE, candidate_size=200)
    elif args.net == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method='hard', device=DEVICE, candidate_size=200)
    elif args.net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net,nms_method='hard', device=DEVICE, candidate_size=200)
    elif args.net == 'mb2-ssd-lite' or args.net == "mb3-large-ssd-lite" or args.net == "mb3-small-ssd-lite":
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method='hard', device=DEVICE, candidate_size=200, config=config)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    totalsum = 0
    totalsumtarget = 0
    totalsumtext = 0

    totalcnt = 0
    totaltargetcnt = 0
    totaltextcnt = 0

    matchcnt = 0
    matchtargetcnt = 0
    matchtextcnt = 0

    facnt = 0

    try:
        for i in range(len(dataset)):
            image = dataset.get_image(i)
            a, gtbox, gtlabel = dataset.__getitem__(i)

            currcnt = gtbox.shape[0]
            currtargetcnt = np.count_nonzero(gtlabel==1)
            currtextcnt = np.count_nonzero(gtlabel==2)
            totalcnt = totalcnt + gtbox.shape[0]
            totaltargetcnt = totaltargetcnt + currtargetcnt
            totaltextcnt = totaltextcnt + currtextcnt

            gtboxes = torch.tensor(gtbox)
            boxes, labels, probs = predictor.predict(image, 20, iou_treshold)
            sum = 0
            sumtarget = 0
            sumtext = 0
            targetcnt = 0
            textcnt = 0

            predcnt = list(boxes.size())[0]
            currmatchcnt = 0
            currmatchtargetcnt = 0
            currmatchtextcnt = 0

            currfacnt = 0

            for j in range(gtboxes.size(0)):
                iou = box_utils.iou_of(gtboxes[j], boxes)
                maxval = torch.max(iou)
                xor = 1 - maxval
                sum = sum + xor

                if gtlabel[j] == 1:
                    sumtarget = sumtarget + xor
                    targetcnt = targetcnt + 1
                elif gtlabel[j] == 2:
                    sumtext = sumtext + xor
                    textcnt = textcnt + 1

                if maxval > iou_treshold:
                    currmatchcnt = currmatchcnt + 1

                    if gtlabel[j] == 1:
                        currmatchtargetcnt = currmatchtargetcnt + 1
                    elif gtlabel[j] == 2:
                        currmatchtextcnt = currmatchtextcnt + 1

            totalsum = totalsum + sum / gtboxes.size(0)
            totalsumtarget = totalsumtarget + sumtarget / targetcnt
            totalsumtext = totalsumtext + sumtext / textcnt

            matchcnt = matchcnt + currmatchcnt
            matchtargetcnt = matchtargetcnt + currmatchtargetcnt
            matchtextcnt = matchtextcnt + currmatchtextcnt

            facheck = list(probs > iou_treshold).count(True) - gtboxes.size(0)

            if facheck > 0:
                facnt = facnt + facheck

        retavr = (totalsum/len(dataset)).item()
        retavrtarget = (totalsumtarget/len(dataset)).item()
        retavrtext = (totalsumtext/len(dataset)).item()

        rettotalap = matchcnt/totalcnt
        rettotaltargetap = matchtargetcnt/totaltargetcnt
        rettotaltextap = matchtextcnt/totaltextcnt
        retfacnt = facnt

    except:
        retavr = 1.0
        retavrtarget = 1.0
        retavrtext = 1.0

        rettotalap = 0
        rettotaltargetap = 0
        rettotaltextap = 0
        retfacnt = 0

    return retavr, retavrtarget, retavrtext, rettotalap, rettotaltargetap, rettotaltextap, retfacnt

if __name__ == '__main__':
    timer = Timer()

    best = 'best.pt'
    last = 'last.pt'
    best_loss = 100.0

    logging.info(args)
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-large-ssd-lite':
        create_net = lambda num: create_mobilenetv3_large_ssd_lite(num)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-small-ssd-lite':
        if args.image_size == 240:
            config = mobilenetv3_ssd_config_240
        elif args.image_size == 200:
            config = mobilenetv3_ssd_config_200
        elif args.image_size == 160:
            config = mobilenetv3_ssd_config_160
        else:
            config = mobilenetv1_ssd_config

        create_net = lambda num: create_mobilenetv3_small_ssd_lite(num, config=config)
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    logging.info(f"imagesize!!!! {config.image_size}")
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    #train_transform = None
    #target_transform = None
    #test_transform = None

    logging.info("Prepare training datasets.")
    if args.dataset_type == 'voc':
        dataset = VOCDataset(args.datasets, transform=train_transform,
                             target_transform=target_transform)
        label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
        store_labels(label_file, dataset.class_names)
        num_classes = len(dataset.class_names)
    elif args.dataset_type == 'open_images':
        dataset = OpenImagesDataset(args.datasets,
             transform=train_transform, target_transform=target_transform,
             dataset_type="train", balance_data=args.balance_data)
        label_file = os.path.join(args.checkpoint_folder, "open-images-model-labels.txt")
        store_labels(label_file, dataset.class_names)
        logging.info(dataset)
        num_classes = len(dataset.class_names)

    else:
        raise ValueError(f"Dataset type {args.dataset_type} is not supported.")

    logging.info(f"Stored labels into file {label_file}.")
#    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(dataset)))
    train_loader = DataLoader(dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    logging.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif args.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(args.datasets,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
        logging.info(val_dataset)

    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    logging.info("Build network.")
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    logging.info(f"Init Loss Function: {args.lossfunc}")
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE, losstype=args.lossfunc)
#    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
#                                weight_decay=args.weight_decay)
    optimizer = RAdam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.pretrained_ssd:
        loadnet = torch.load(args.pretrained_ssd)
        optimizer.load_state_dict(loadnet['optimizer_state_dict'])

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

#    if args.pretrained_ssd:
#        loadnet = torch.load(args.pretrained_ssd)
#        scheduler.load_state_dict(loadnet['scheduler_state_dict'])

    loglen = 0
    try:
        loglen = len(os.listdir(args.checkpoint_folder))
        targetPath = args.checkpoint_folder + '/train' + str(loglen)
        os.mkdir(targetPath)
    except:
        loglen = 0
        targetPath = args.checkpoint_folder + '/train' + str(loglen)
        os.mkdir(targetPath)

    print('targetPath: ', targetPath)

    tb_writer = None
    tb_writer = SummaryWriter(targetPath)  # Tensorboard

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        train_loss, train_regression_loss, train_classification_loss = train(train_loader, net, criterion, optimizer,
              device=DEVICE, epoch=epoch, tb_writer=tb_writer)
        scheduler.step()
        logging.info(
            f"Epoch: {epoch}, " +
            f"Train Loss: {train_loss:.4f}, " +
            f"Train Regression Loss {train_regression_loss:.4f}, " +
            f"Train Classification Loss: {train_classification_loss:.4f}"
        )

        val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
        logging.info(
            f"Validation Loss: {val_loss:.4f}, " +
            f"Validation Regression Loss {val_regression_loss:.4f}, " +
            f"Validation Classification Loss: {val_classification_loss:.4f}"
        )

        cname, cap = eval(args, net.state_dict(), DEVICE, 0.5, label_file, targetPath, config)
        logging.info(
            f"map: {sum(cap)/len(cap):.4f}, " +
            f"{cname[1]}: {cap[0]:.4f}, " +
            f"{cname[2]}: {cap[1]:.4f}"
        )

        totalavr, totalavrtarget, totalavrtext, totalap, totaltargetap, totaltextap, facnt = cal_boxdiff(args, net.state_dict(), DEVICE, 0.5, label_file, config)
        logging.info(
            f"totalavr: {totalavr}, " +
            f"totalavrtarget: {totalavrtarget}, " +
            f"totalavrtext: {totalavrtext}, " +
            f"totalap: {totalap}, " +
            f"totaltargetap: {totaltargetap}, " +
            f"totaltextap: {totaltextap}, " +
            f"facnt: {facnt}"
        )

        model_path = targetPath + '/' + args.net + '-' + last

        #torch.save(net.state_dict(), model_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_regression_loss': val_regression_loss,
            'val_classification_loss': val_classification_loss
            }, model_path)

        if best_loss > val_loss:
          best_loss = val_loss

        if best_loss == val_loss:
          model_path = targetPath + '/' + args.net + "-" + best
          #torch.save(net.state_dict(), model_path)
          torch.save({
              'epoch': epoch,
              'model_state_dict': net.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'scheduler_state_dict': scheduler.state_dict(),
              'val_regression_loss': val_regression_loss,
              'val_classification_loss': val_classification_loss
              }, model_path)
          logging.info(f"Saved model {model_path}")

        if tb_writer:
          tb_writer.add_scalar('train/loss', train_loss, epoch)
          tb_writer.add_scalar('train/regression_loss', train_regression_loss, epoch)
          tb_writer.add_scalar('train/classification_loss', train_classification_loss, epoch)
          tb_writer.add_scalar('val/loss', val_loss, epoch)
          tb_writer.add_scalar('val/regression_loss', val_regression_loss, epoch)
          tb_writer.add_scalar('val/classification_loss', val_classification_loss, epoch)
          tb_writer.add_scalar('val/map', sum(cap)/len(cap), epoch)
          tb_writer.add_scalar('val/target', cap[0], epoch)
          tb_writer.add_scalar('val/text', cap[1], epoch)
          tb_writer.add_scalar('box/total', totalavr, epoch)
          tb_writer.add_scalar('box/target', totalavrtarget, epoch)
          tb_writer.add_scalar('box/text', totalavrtext, epoch)
          tb_writer.add_scalar('box/totalap', totalap, epoch)
          tb_writer.add_scalar('box/totaltargetap', totaltargetap, epoch)
          tb_writer.add_scalar('box/totaltextap', totaltextap, epoch)
          tb_writer.add_scalar('box/facnt', facnt, epoch)
