import argparse
import time
import torch
import sys
from torch.utils.data import DataLoader

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from datasets import FCVID, miniKINETICS, ACTNET
from model import ModelGCNConcAfter as Model

parser = argparse.ArgumentParser(description='GCN Video Classification')
parser.add_argument('model', nargs=1, help='trained model')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='actnet', choices=['fcvid', 'minikinetics', 'actnet'])
parser.add_argument('--dataset_root', default='/home/dimidask/Projects/ActivityNet120', help='dataset root directory')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_objects', type=int, default=50, help='number of objects with best DoC')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--ext_method', default='VIT', choices=['VIT', 'RESNET'], help='Extraction method for features')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
parser.add_argument('--frames', type=int, default=5, help='Number of frames for Metrics')
args = parser.parse_args()


def metrics_run(model,  dataset, loader, scores, scores_bestframes, scores_worstframes, device):
    gidx = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            feats, feat_global, _, _ = batch

            # Run model with all frames
            feats = feats.to(device)
            feat_global = feat_global.to(device)

            out_data, _, wids_frame_local, wids_frame_global = model(feats, feat_global, device, get_adj=True)

            shape = out_data.shape[0]

            # Choose Best and Worst Frames
            average_wids = np.mean(np.array([wids_frame_local, wids_frame_global]), axis=0)
            index_bestframes = torch.tensor(np.sort(np.argsort(average_wids, axis=1)[:, -args.frames:])).to(device)
            index_worstframes = torch.tensor(np.sort(np.argsort(average_wids, axis=1)[:, :-args.frames])).to(device)

            feats_worstframes = feats.gather(dim=1, index=index_worstframes.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dataset.NUM_BOXES, dataset.NUM_FEATS)).to(device)
            feat_global_worstframes = feat_global.gather(dim=1, index=index_worstframes.unsqueeze(-1).expand(-1, -1, dataset.NUM_FEATS)).to(device)

            feats_bestframes = feats.gather(dim=1, index=index_bestframes.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dataset.NUM_BOXES, dataset.NUM_FEATS)).to(device)
            feat_global_bestframes = feat_global.gather(dim=1, index=index_bestframes.unsqueeze(-1).expand(-1, -1, dataset.NUM_FEATS)).to(device)

            out_data_bestframes = model(feats_bestframes, feat_global_bestframes, device)
            out_data_worstframes = model(feats_worstframes, feat_global_worstframes, device)

            scores[gidx:gidx+shape, :] = out_data.cpu()
            scores_bestframes[gidx:gidx + shape, :] = out_data_bestframes.cpu()
            scores_worstframes[gidx:gidx + shape, :] = out_data_worstframes.cpu()

            gidx += shape


def main():
    if args.dataset == 'fcvid':
        dataset = FCVID(args.dataset_root, is_train=False, ext_method=args.ext_method)
    elif args.dataset == 'actnet':
        dataset = ACTNET(args.dataset_root, is_train=False, ext_method=args.ext_method)
    elif args.dataset == 'minikinetics':
        dataset = miniKINETICS(args.dataset_root, is_train=False, ext_method=args.ext_method)
    else:
        sys.exit("Unknown dataset!")
    device = torch.device('cuda:0')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    model = Model(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS).to(device)
    data = torch.load(args.model[0])
    model.load_state_dict(data['model_state_dict'])

    num_test = len(dataset)
    scores = torch.zeros((num_test, dataset.NUM_CLASS), dtype=torch.float32)
    scores_bestframes = torch.zeros((num_test, dataset.NUM_CLASS), dtype=torch.float32)
    scores_worstframes = torch.zeros((num_test, dataset.NUM_CLASS), dtype=torch.float32)

    metrics_run(model, dataset, loader, scores, scores_bestframes, scores_worstframes, device)

    # Compute and Print Metrics
    computemetrics(scores, scores_bestframes, scores_worstframes, dataset.labels)


def increaseconfclass(scores_y, scores_o):
    videos = len(scores_y)
    conf = 0
    for video in range(videos):
        if scores_y[video] < scores_o[video]:
            conf += 1
    return (conf / videos) * 100


def averagedropclass(scores_y, scores_o):
    videos = len(scores_y)
    drop = 0
    for video in range(videos):
        drop += max(0, scores_y[video] - scores_o[video]) / scores_y[video]
    return (drop / videos) * 100


def fidelityminus(scores, scores_bestframes, labels):
    videos = len(scores)
    count = 0.
    num_labels = 0.
    for video in range(videos):
        top = np.argwhere(labels[video] == np.max(labels[video]))
        top = top.tolist()
        num_top = len(top)
        num_labels += num_top
        if num_top == 1:
            scores_label = np.argmax(scores[video])
            scores_bestframes_label = np.argmax(scores_bestframes[video])
            first = 1. if (scores_label in top) else 0.
            second = 1. if (scores_bestframes_label in top) else 0.
            count += first - second
        else:
            scores_label = np.sort(np.argsort(scores[video])[-num_top:])
            scores_bestframes_label = np.sort(np.argsort(scores_bestframes[video])[-num_top:])
            for i in range(num_top):
                first = 1. if (scores_label[i] in top) else 0.
                second = 1. if (scores_bestframes_label[i] in top) else 0.
                count += first - second
    return count / num_labels


def fidelityplus(scores, scores_worstframes, labels):
    videos = len(scores)
    count = 0.
    num_labels = 0.
    for video in range(videos):
        top = np.argwhere(labels[video] == np.max(labels[video]))
        top = top.tolist()
        num_top = len(top)
        num_labels += num_top
        if num_top == 1:
            scores_label = np.argmax(scores[video])
            scores_worstframes_label = np.argmax(scores_worstframes[video])
            first = 1. if (scores_label in top) else 0.
            second = 1. if (scores_worstframes_label in top) else 0.
            count += first - second
        else:
            scores_label = np.sort(np.argsort(scores[video])[-num_top:])
            scores_worstframes_label = np.sort(np.argsort(scores_worstframes[video])[-num_top:])
            for i in range(num_top):
                first = 1. if (scores_label[i] in top) else 0.
                second = 1. if (scores_worstframes_label[i] in top) else 0.
                count += first - second
    return count / num_labels


def computemetrics(scores, scores_bestframes, scores_worstframes, labels):
    # Softmax Classifier
    classify = nn.Sigmoid()
    scores = classify(scores)
    scores_bestframes = classify(scores_bestframes)
    scores_worstframes = classify(scores_worstframes)

    # Find accepted class and select only those from scores
    classid = torch.argmax(scores, dim=1)
    scoresnew = torch.squeeze(scores.gather(dim=1, index=classid.unsqueeze(-1)))
    scores_bestframes_new = torch.squeeze(scores_bestframes.gather(dim=1, index=classid.unsqueeze(-1)))

    # Compute Metrics
    average_drop_best = averagedropclass(scoresnew, scores_bestframes_new)
    increase_conf_best = increaseconfclass(scoresnew, scores_bestframes_new)
    print("Average Drop: {:.2f} %".format(average_drop_best))
    print("Increase in Confidence: {:.2f} %".format(increase_conf_best))

    # Change tensors to 1d-arrays
    scores = scores.numpy()
    scores_bestframes = scores_bestframes.numpy()
    scores_worstframes = scores_worstframes.numpy()

    fidelitym = fidelityminus(scores, scores_bestframes, labels)
    fidelityp = fidelityplus(scores, scores_worstframes, labels)

    print("Fidelity Plus: {:.2f} %".format(fidelityp * 100))
    print("Fidelity Minus: {:.2f} %".format(fidelitym * 100))


if __name__ == '__main__':
    main()
