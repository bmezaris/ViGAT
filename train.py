import argparse
import time
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import FCVID, miniKINETICS, ACTNET
from model import ModelGCNConcAfter as Model

parser = argparse.ArgumentParser(description='GCN Video Classification')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='actnet', choices=['fcvid', 'minikinetics', 'actnet'])
parser.add_argument('--dataset_root', default='/home/dimidask/Projects/ActivityNet120', help='dataset root directory')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--milestones', nargs="+", type=int, default=[110, 160], help='milestones of learning decay')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_objects', type=int, default=50, help='number of objects with best DoC')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--ext_method', default='VIT', choices=['VIT', 'RESNET'], help='Extraction method for features')
parser.add_argument('--resume', default=None, help='checkpoint to resume training')
parser.add_argument('--save_interval', type=int, default=10, help='interval for saving models (epochs)')
parser.add_argument('--save_folder', default='weights', help='directory to save checkpoints')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
args = parser.parse_args()


def train(model, loader, crit, opt, sched, device):
    epoch_loss = 0
    for i, batch in enumerate(loader):
        feats, feat_global, label, _ = batch

        feats = feats.to(device)
        feat_global = feat_global.to(device)
        label = label.to(device)

        opt.zero_grad()
        out_data = model(feats, feat_global, device)
        loss = crit(out_data, label)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()

    sched.step()
    return epoch_loss / len(loader)


def main():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if args.dataset == 'fcvid':
        dataset = FCVID(args.dataset_root, is_train=True, ext_method=args.ext_method)
        crit = nn.BCEWithLogitsLoss()
    elif args.dataset == 'actnet':
        dataset = ACTNET(args.dataset_root, is_train=True, ext_method=args.ext_method)
        crit = nn.BCEWithLogitsLoss()
    elif args.dataset == 'minikinetics':
        dataset = miniKINETICS(args.dataset_root, is_train=True, ext_method=args.ext_method)
        crit = nn.CrossEntropyLoss()
    else:
        sys.exit("Unknown dataset!")

    device = torch.device('cuda:0')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    if args.verbose:
        print("running on {}".format(device))
        print("num samples={}".format(len(dataset)))
        print("missing videos={}".format(dataset.num_missing))

    start_epoch = 0
    model = Model(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones)
    if args.resume:
        data = torch.load(args.resume)
        start_epoch = data['epoch']
        model.load_state_dict(data['model_state_dict'])
        opt.load_state_dict(data['opt_state_dict'])
        sched.load_state_dict(data['sched_state_dict'])
        if args.verbose:
            print("resuming from epoch {}".format(start_epoch))

    model.train()
    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.perf_counter()
        loss = train(model, loader, crit, opt, sched, device)
        t1 = time.perf_counter()

        if (epoch + 1) % args.save_interval == 0:
            sfnametmpl = 'model-{}-{:03d}.pt'
            sfname = sfnametmpl.format(args.dataset, epoch + 1)
            spth = os.path.join(args.save_folder, sfname)
            torch.save({
                'epoch': epoch + 1,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'sched_state_dict': sched.state_dict()
            }, spth)
        if args.verbose:
            print("[epoch {}] loss={} dt={:.2f}sec".format(epoch + 1, loss, t1 - t0))


if __name__ == '__main__':
    main()
