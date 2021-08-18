from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.utils.misc import Timer
import cv2
import sys
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
import torch
import numpy as np
import argparse
from vision.ssd.config import mobilenetv1_ssd_config, mobilenetv3_ssd_config_240, mobilenetv3_ssd_config_200, mobilenetv3_ssd_config_160


parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)

parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument('--image_size', default=300, type=int, choices=[300, 240, 200, 160],
                    help='Input Image size')
args = parser.parse_args()


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net_type = args.net
model_path = args.trained_model
label_path = args.label_file

def cal_boxdiff(predictor, dataset, iou_threshold):
    totalsum = 0
    totalsumtarget = 0
    totalsumtext = 0

    try:
        for i in range(len(dataset)):
            image = dataset.get_image(i)
            a, gtbox, gtlabel = dataset.__getitem__(i)
            gtboxes = torch.tensor(gtbox)
            boxes, labels, probs = predictor.predict(image, -1, iou_threshold)
            sum = 0
            sumtarget = 0
            sumtext = 0
            targetcnt = 0
            textcnt = 0

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

            totalsum = totalsum + sum / gtboxes.size(0)
            totalsumtarget = totalsumtarget + sumtarget / targetcnt
            totalsumtext = totalsumtext + sumtext / textcnt

        retavr = (totalsum/len(dataset)).item()
        retavrtarget = (totalsumtarget/len(dataset)).item()
        retavrtext = (totalsumtext/len(dataset)).item()
    except:
        retavr = 1.0
        retavrtarget = 1.0
        retavrtext = 1.0

    return retavr, retavrtarget, retavrtext

def cal_boxdiff2(args, net_state_dict, DEVICE, iou_threshold, label_file):
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
        net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb3-small-ssd-lite':
        if args.image_size == 240:
            config = mobilenetv3_ssd_config_240
        elif args.image_size == 200:
            config = mobilenetv3_ssd_config_200
        elif args.image_size == 160:
            config = mobilenetv3_ssd_config_160
        else:
            config = mobilenetv1_ssd_config

        net = create_mobilenetv3_small_ssd_lite(len(class_names), config=config, is_test=True)
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
        #predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method='hard', device=DEVICE, candidate_size=200)
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=DEVICE, config=config)
        image = dataset.get_image(0)
        boxes, labels, probs = predictor.predict(image, -1, iou_threshold)
        print(boxes)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    totalsum = 0
    totalsumtarget = 0
    totalsumtext = 0

    for i in range(len(dataset)):
        image = dataset.get_image(i)
        a, gtbox, gtlabel = dataset.__getitem__(i)
        gtboxes = torch.tensor(gtbox)
        boxes, labels, probs = predictor.predict(image, -1, iou_threshold)
        print(gtboxes)
        print(boxes)
        sum = 0
        sumtarget = 0
        sumtext = 0
        targetcnt = 0
        textcnt = 0

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

        totalsum = totalsum + sum / gtboxes.size(0)
        totalsumtarget = totalsumtarget + sumtarget / targetcnt
        totalsumtext = totalsumtext + sumtext / textcnt

    retavr = (totalsum/len(dataset)).item()
    retavrtarget = (totalsumtarget/len(dataset)).item()
    retavrtext = (totalsumtext/len(dataset)).item()

    return retavr, retavrtarget, retavrtext

def tfpercent(predictor, dataset, iou_threshold):
    totalcnt = 0
    totaltargetcnt = 0
    totaltextcnt = 0

    matchcnt = 0
    matchtargetcnt = 0
    matchtextcnt = 0

    facnt = 0

#    try:
    for i in range(len(dataset)):
#    for i in range(1):
        image = dataset.get_image(i)
        a, gtbox, gtlabel = dataset.__getitem__(i)

        currcnt = gtbox.shape[0]
        currtargetcnt = np.count_nonzero(gtlabel==1)
        currtextcnt = np.count_nonzero(gtlabel==2)
        totalcnt = totalcnt + gtbox.shape[0]
        totaltargetcnt = totaltargetcnt + currtargetcnt
        totaltextcnt = totaltextcnt + currtextcnt

        gtboxes = torch.tensor(gtbox)
        boxes, labels, probs = predictor.predict(image, -1, iou_threshold)

        predcnt = list(boxes.size())[0]
        currmatchcnt = 0
        currmatchtargetcnt = 0
        currmatchtextcnt = 0

        currfacnt = 0

        for j in range(gtboxes.size(0)):
            iou = box_utils.iou_of(gtboxes[j], boxes)
            maxval = torch.max(iou)

            if maxval > iou_threshold:
                currmatchcnt = currmatchcnt + 1

                if gtlabel[j] == 1:
                    currmatchtargetcnt = currmatchtargetcnt + 1
                elif gtlabel[j] == 2:
                    currmatchtextcnt = currmatchtextcnt + 1

        matchcnt = matchcnt + currmatchcnt
        matchtargetcnt = matchtargetcnt + currmatchtargetcnt
        matchtextcnt = matchtextcnt + currmatchtextcnt

        facheck = list(probs > iou_threshold).count(True) - gtboxes.size(0)

        if facheck > 0:
            facnt = facnt + facheck

#    except:
#        retavr = 1.0
#        retavrtarget = 1.0
#        retavrtext = 1.0
#        print(totalcnt, totaltargetcnt, totaltextcnt, matchcnt, matchtargetcnt, matchtextcnt, facnt)
    print(totalcnt, totaltargetcnt, totaltextcnt, matchcnt, matchtargetcnt, matchtextcnt, facnt)
    return matchcnt/totalcnt, matchtargetcnt/totaltargetcnt, matchtextcnt/totaltextcnt, facnt

class_names = [name.strip() for name in open(label_path).readlines()]

if args.image_size == 240:
    config = mobilenetv3_ssd_config_240
elif args.image_size == 200:
    config = mobilenetv3_ssd_config_200
elif args.image_size == 160:
    config = mobilenetv3_ssd_config_160
else:
    config = mobilenetv1_ssd_config

net = create_mobilenetv3_small_ssd_lite(len(class_names), config=config, is_test=True)
net.load(model_path)
net.to(DEVICE)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=DEVICE, config=config)

#config = mobilenetv1_ssd_config
#test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
#target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)

#val_dataset = OpenImagesDataset(args.datasets, transform=test_transform, target_transform=target_transform, dataset_type="test")
#val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False)
#dataset = OpenImagesDataset(args.datasets, dataset_type="test")
dataset = OpenImagesDataset('/content/dataset', dataset_type="test")

print(cal_boxdiff(predictor, dataset, args.iou_threshold))
print(tfpercent(predictor, dataset, args.iou_threshold))

#from types import SimpleNamespace as sn

#dictargs = {
#        'net': 'mb3-small-ssd-lite',
#        'datasets': '/content/dataset'
#    }
#args = sn(**dictargs)
#print(args.net)
#print(args.datasets)

#model = torch.load(model_path, map_location=lambda storage, loc: storage)
#model = torch.load(model_path)
#net.load_state_dict(model['model_state_dict'])
#print(cal_boxdiff2(args, model['model_state_dict'], 'cuda:0', 0.5, label_path))
#for i in range(len(dataset)):
#    image = dataset.get_image(i)
#totalsum = 0
#totalsumtarget = 0
#totalsumtext = 0
#
#for i in range(len(dataset)):
#    image = dataset.get_image(i)
#    a, gtbox, gtlabel = dataset.__getitem__(i)
#    gtboxes = torch.tensor(gtbox)
#    print(gtboxes)
#    print(gtboxes.size())
#    print(gtboxes[0])
#    orig_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#    boxes, labels, probs = predictor.predict(image, 10, 0.4)
#    print(boxes)
#    print(gtboxes.size(0))
#    sum = 0
#    sumtarget = 0
#    sumtext = 0
#    targetcnt = 0
#    textcnt = 0
#    print(gtlabel)
#
#    for j in range(gtboxes.size(0)):
#        iou = box_utils.iou_of(gtboxes[j], boxes)
#        maxval = torch.max(iou)
#        xor = 1 - maxval
#        sum = sum + xor
#
#        if gtlabel[j] == 1:
#            sumtarget = sumtarget + xor
#            targetcnt = targetcnt + 1
#        elif gtlabel[j] == 2:
#            sumtext = sumtext + xor
#            textcnt = textcnt + 1
#
#    totalsum = totalsum + sum / gtboxes.size(0)
#    totalsumtarget = totalsumtarget + sumtarget / targetcnt
#    totalsumtext = totalsumtext + sumtext / textcnt
#
#print((totalsum/len(dataset)).item())
#print((totalsumtarget/len(dataset)).item())
#print((totalsumtext/len(dataset)).item())
#
#orig_image = cv2.imread(image_path)
#image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
#boxes, labels, probs = predictor.predict(image, 10, 0.4)
#
#for i in range(boxes.size(0)):
#    box = boxes[i, :]
#    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
#    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
#    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
#    cv2.putText(orig_image, label,
#                (box[0] + 20, box[1] + 40),
#                cv2.FONT_HERSHEY_SIMPLEX,
#                1,  # font scale
#                (255, 0, 255),
#                2)  # line type
#path = "run_ssd_example_output.jpg"
#cv2.imwrite(path, orig_image)
#print(f"Found {len(probs)} objects. The output image is {path}")
