import os
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DistributedSampler,BatchSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from annotations import build_vocab,get_info
from dataset_build import AGDataset_train
from model import sgODE_model
from discriminator import Critic,obj_Discriminator
from utils import gradient_penalty,compute_losses

parser = argparse.ArgumentParser()

#path
parser.add_argument('--frames_path', default='/data/scene_understanding/action_genome/frames/')
parser.add_argument('--annotations_path', default='/data/scene_understanding/action_genome/annotations/')
parser.add_argument('--checkpoint_path',default='./')
parser.add_argument('--generate_frame_path',default='./generation/')

#neural network set
parser.add_argument('--image_size', default=(64,64))
parser.add_argument('--nepoch',default=20)
parser.add_argument('--lr_G', type=float, default=1e-5)
parser.add_argument('--lr_D', type=float, default=1e-5)
parser.add_argument('--lr_D_obj', type=float, default=1e-5)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--mask_size', default=16, type=int)
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--refinement_network_dims', default=(1024,512,256,128,64))
parser.add_argument('--layout_noise_dim', default=64, type=int)
parser.add_argument('--d_padding', default='valid')
parser.add_argument('--object_size', default=32, type=int)

#Loss function set
parser.add_argument('--bbox_loss_weight',default=1.0,type=float)
parser.add_argument('--img_loss_weight',default=1.0,type=float)
parser.add_argument('--pb_loss_weight',default=0.5,type=float)
parser.add_argument('--ac_loss_weight',default=0.1,type=float)
parser.add_argument('--C_Lambda',default=10)
parser.add_argument('--crit_repeats',default=5)

#ODE set
parser.add_argument('--tol', type=float, default=1e-3)#tolerance
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--time_step',type=float,default=0.01)

#other set
parser.add_argument('--nonperson_filer',default=True)
parser.add_argument('--gpu', default=0,type=int)
parser.add_argument('--load_model',default=False)
parser.add_argument('--savefig_path',default='./')
parser.add_argument('--Multi_GPU',default=False)
parser.add_argument('--Train',default=True)

args = parser.parse_args()

def save_checkpoint(state,args,epoch):
    print('Saving checkpoint in epoch %d'%epoch)
    file=args.checkpoint_path+'my_checkpoint_SG_700_%d.pth.tar'%epoch
    torch.save(state,file)

def load_checkpoint(checkpoint,netG,netD,optimizer_netG,optimizer_netD):
    print('Loading checkpoint')
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizer_netG.load_state_dict(checkpoint['optimizer_netG'])
    optimizer_netD.load_state_dict(checkpoint['optimizer_netD'])


def train(args):
    vocab = build_vocab(args.annotations_path)
    kf_info, video_info, train_list, test_list, max_length = get_info(args.annotations_path, args.nonperson_filer,
                                                                      vocab)

    dset_train_kwargs = {
        'vocab': vocab,
        'video_info': video_info,
        'frames_path': args.frames_path,
        'image_size': args.image_size,
        'train_list': train_list,
    }
    dataset_train = AGDataset_train(**dset_train_kwargs)

    if args.Multi_GPU == True:
        # if use slurm
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
            print(args.rank, args.world_size, args.gpu)
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)

        # initialize the group
        torch.distributed.init_process_group(backend='nccl',
                                             world_size=args.world_size,
                                             rank=args.rank)
        torch.distributed.barrier()
        device = torch.device('cuda')

        sampler = DistributedSampler(dataset_train)
        batch_sampler = BatchSampler(sampler, 1, drop_last=True)
        train_loader = DataLoader(dataset=dataset_train, batch_sampler=batch_sampler)

        netG = sgODE_model(args, vocab, device).to(device)
        netD = Critic(im_chan=3, hidden_dim=64).to(device)
        netD_obj = obj_Discriminator(args, vocab).to(device)

        netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=[args.gpu])
        netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[args.gpu])
        netD_obj = torch.nn.parallel.DistributedDataParallel(netD_obj, device_ids=[args.gpu])

    else:  # 1 GPU
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)

        netG = sgODE_model(args, vocab, device).to(device)
        netD = Critic(im_chan=3, hidden_dim=64).to(device)
        netD_obj = obj_Discriminator(args, vocab).to(device)

    netG.train()
    netD.train()
    netD_obj.train()

    optimizer_netG = optim.Adam(netG.parameters(), lr=args.lr_G)
    optimizer_netD = optim.Adam(netD.parameters(), lr=args.lr_D)
    optimizer_netD_obj = optim.Adam(netD.parameters(), lr=args.lr_D_obj)

    criterion = nn.BCELoss().to(device)

    if args.load_model:
        load_checkpoint(torch.load(args.checkpoint_path + 'my_checkpoint.pth.tar'), netG,
                        netD, optimizer_netG, optimizer_netD)


    for epoch in range(args.nepoch):
        if args.Multi_GPU == True:
            sampler.set_epoch(epoch)

        for batch_idx, (
        video_name, frames_num, key_frame, frame_name, key_frame_inter, obj_list, triple_list, bbox_list) in enumerate(
                train_loader):

            video_name = video_name[0]
            frames_num = int(frames_num)

            for i in range(0, frames_num):
                if i == frames_num - 1:
                    continue
                else:
                    obj = obj_list[i].squeeze(0).to(device)
                    triple = triple_list[i].squeeze(0).to(device)
                    bbox_gt = bbox_list[i].squeeze(0).to(device)

                    inter = key_frame_inter[i]

                    video_start_time = int(inter[0])
                    start = int(inter[0])
                    end = int(inter[1])

                    if i == 0:
                        img = key_frame[i].to(device)
                    else:
                        img = frame_fake[-1].unsqueeze(dim=0)

                    start_time = start - video_start_time
                    end_time = end - video_start_time

                    steps = (end_time - start_time) if (end_time - start_time) < 10 else 10

                    time_stamps = torch.linspace(start_time * 0.01, end_time * 0.01, steps=steps + 1).float().to(device)

                    frame_real = dataset_train.get_real_frame(start, end, video_name).to(device)

                    mean_iteration_critic_loss = 0

                    for c in range(args.crit_repeats):
                        frame_pred, boxes_pred = netG(img, obj, triple, obj_to_img=None, boxes_gt=bbox_gt,
                                                      masks_gt=None,
                                                      time_stamps=time_stamps)

                        frame_fake = frame_pred.detach()

                        optimizer_netD.zero_grad()
                        crit_fake = netD(frame_fake)
                        crit_real = netD(frame_real)

                        epsilon = torch.rand(len(frame_real), 1, 1, 1, device=device, requires_grad=True)
                        gp = gradient_penalty(netD, frame_real, frame_fake, epsilon)
                        loss_netD = -(torch.mean(crit_real) - torch.mean(crit_fake)) + args.C_Lambda * gp
                        mean_iteration_critic_loss += loss_netD.detach().item() / args.crit_repeats
                        loss_netD.backward(retain_graph=True)
                        optimizer_netD.step()

                    optimizer_netG.zero_grad()
                    loss_netG1 = compute_losses(args, boxes_pred, bbox_gt, frame_pred, frame_real)
                    crit_fake_G = netD(frame_pred)
                    loss_netG2 = -torch.mean(crit_fake_G)
                    loss_netG = loss_netG1 + loss_netG2
                    loss_netG.backward()
                    optimizer_netG.step()

                    frame_fake = frame_pred.detach()
                    scores_fake, ac_loss_fake = netD_obj(frame_fake[0].unsqueeze(dim=0), obj, bbox_gt)
                    scores_real, ac_loss_real = netD_obj(frame_real[0].unsqueeze(dim=0), obj, bbox_gt)

                    disc_fake = scores_fake.reshape(-1)
                    lossD_obj1 = criterion(disc_fake, torch.zeros_like(disc_fake))
                    disc_real = scores_real.reshape(-1)
                    lossD_obj2 = criterion(disc_real, torch.ones_like(disc_real))
                    lossD_obj = (lossD_obj1 + lossD_obj2) / 2 + ac_loss_fake + ac_loss_real

                    optimizer_netD_obj.zero_grad()
                    lossD_obj.backward()
                    optimizer_netD_obj.step()

                    # step += 1
                    # writer.add_scalar('LossG per KF', loss_netG, global_step=step)
                    # writer.add_scalar('LossD per KF', mean_iteration_critic_loss, global_step=step)

        # if epoch % 1 == 0:
        #     checkpoint = {'netG_state_dict': netG.state_dict(), 'netD_state_dict': netD.state_dict(),
        #                   'optimizer_netG': optimizer_netG.state_dict(), 'optimizer_netD': optimizer_netD.state_dict()}
        #
        #     save_checkpoint(checkpoint, args, epoch)

if __name__ == '__main__':
    train(args)






