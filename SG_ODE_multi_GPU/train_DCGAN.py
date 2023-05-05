import os
import argparse

from imageio import imwrite

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DistributedSampler,BatchSampler

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from annotations import build_vocab,get_info
from dataset_build import AGDataset_train
from model_pb import sgODE_model,Discriminator,compute_losses
from utils import imagenet_deprocess_batch

parser = argparse.ArgumentParser()

#path
parser.add_argument('--frames_path', default='F:/dataset/Action Genome/frames/')
parser.add_argument('--annotations_path', default='F:/dataset/Action Genome/Annotations v1.0/')
parser.add_argument('--csv_path',default='F:/dataset/vocab_kf_info/')
# parser.add_argument('--frames_path', default='/data/scene_understanding/action_genome/frames/')
# parser.add_argument('--annotations_path', default='/data/scene_understanding/action_genome/annotations/')
parser.add_argument('--checkpoint_path',default='./')
parser.add_argument('--generate_frame_path',default='./generation_SG_multi/')

#neural network set
parser.add_argument('--image_size', default=(64,64))
parser.add_argument('--nepoch',default=20)
# parser.add_argument('--video_num_iterations',default=5)#number of the video for iterations
parser.add_argument('--lr_G', type=float, default=1e-6)
parser.add_argument('--lr_D', type=float, default=1e-7)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--mask_size', default=16, type=int) # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--refinement_network_dims', default=(1024,512,256,128,64))
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--loader_num_workers', default=0, type=int)
# parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

#netG Loss set
# parser.add_argument('--relationship_loss_weight',default=1.0,type=float)
parser.add_argument('--bbox_loss_weight',default=1.0,type=float)
parser.add_argument('--img_loss_weight',default=1.0,type=float)
# parser.add_argument('--iimg_loss_weight',default=1.0,type=float)
parser.add_argument('--pb_loss_weight',default=0.5,type=float)


#ODE set
parser.add_argument('--tol', type=float, default=1e-3)#tolerance
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--time_step',type=float,default=0.01)


#other set
parser.add_argument('--nonperson_filer',default=True)
parser.add_argument('--gpu', default=0,type=int)
parser.add_argument('--checkpoint_num',default=700,type=int)
parser.add_argument('--load_model',default=True)
parser.add_argument('--savefig_path',default='./')
parser.add_argument('--Multi_GPU',default=True)



args = parser.parse_args()


def save_checkpoint(state,args,epoch):
    print('Saving checkpoint in epoch %d'%epoch)
    file=args.checkpoint_path+'my_checkpoint_tf_700_%d.pth.tar'%epoch
    torch.save(state,file)

def load_checkpoint(checkpoint,netG,netD,optimizer_netG,optimizer_netD):
    print('Loading checkpoint')
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizer_netG.load_state_dict(checkpoint['optimizer_netG'])
    optimizer_netD.load_state_dict(checkpoint['optimizer_netD'])


def main(args):

    # build vocab and key_frames_informations
    vocab = build_vocab(args.annotations_path)
    kf_info, video_info, train_list, test_list, max_length = get_info(args.annotations_path, args.nonperson_filer,vocab)

    # reduce dataset
    train_list = train_list[:700]
    train_list=test_list[0]

    dset_train_kwargs = {
        'vocab': vocab,
        'video_info': video_info,
        'frames_path': args.frames_path,
        'image_size': args.image_size,
        'train_list': train_list,
    }
    dataset_train = AGDataset_train(**dset_train_kwargs)

    if args.Multi_GPU==True:
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
                                             rank=args.rank)  # choose nccl as backend using gpus
        torch.distributed.barrier()  # synchronizes all processes
        device = torch.device('cuda')

        sampler=DistributedSampler(dataset_train)
        batch_sampler=BatchSampler(sampler,1,drop_last=True) # YOUR_BATCHSIZE is the batch size per gpu
        train_loader = DataLoader(dataset=dataset_train, batch_sampler=batch_sampler)

        netG = sgODE_model(args, vocab, device).to(device)
        netD = Discriminator(input_channel=3, hidden=64).to(device)

        netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=[args.gpu])  # wrap the model with DDP
        netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[args.gpu])

    else:#1 GPU
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        train_loader=DataLoader(dataset=dataset_train,batch_size=args.batch_size,shuffle=True)

        netG = sgODE_model(args, vocab, device).to(device)
        netD = Discriminator(input_channel=3, hidden=64).to(device)

    netG.train()#启用BatchNormalization 和 Dropout
    netD.train()

    optimizer_netG = optim.Adam(netG.parameters(), lr=args.lr_G, betas=(0.0, 0.9))
    optimizer_netD = optim.Adam(netD.parameters(), lr=args.lr_D, betas=(0.0, 0.9))

    criterion = nn.BCELoss().to(device)

    G_losses = []
    D_losses = []

    if args.load_model:
        load_checkpoint(torch.load(args.checkpoint_path+'my_checkpoint_700_0.pth.tar'),netG,netD,optimizer_netG,optimizer_netD)

    writer=SummaryWriter(f'./runs/out_TF_multi')
    step = 0

    for epoch in range(0,1):
        if args.Multi_GPU==True:
            sampler.set_epoch(epoch)# make shuffling work properly across multiple epochs.

        for batch_idx,(video_name,frames_num,key_frame,frame_name,key_frame_inter,obj_list,triple_list,bbox_list) in enumerate(train_loader):
        #for video_name,frames_num,key_frame,frame_name,key_frame_inter,obj_list,re_list,bbox_list in train_loader:
            print('generate video Nummber %d'%batch_idx)

            video_name = video_name[0]
            frames_num = int(frames_num)

            save_name_pred=0
            save_name_real=0

            for i in range(0,frames_num):

                if i==frames_num-1:#not using the last Key Frame, because the frame num after the last KF is not sure
                    continue

                else:

                    obj=obj_list[i].squeeze(0).to(device)
                    triple=triple_list[i].squeeze(0).to(device)
                    bbox_gt=bbox_list[i].squeeze(0).to(device)

                    inter = key_frame_inter[i]
                    key_frame_name=frame_name[i][0]

                    video_start_time = int(inter[0])
                    start = int(inter[0])
                    end = int(inter[1])

                    if i == 0:
                        img = key_frame[i].to(device)  # (1,3,64,64)
                    else:
                        img = frame_fake[-1].unsqueeze(dim=0)  # (1,3,64,64)

                    start_time = start - video_start_time
                    end_time = end - video_start_time

                    steps = (end_time - start_time) if (end_time - start_time) < 10 else 10

                    time_stamps = torch.linspace(start_time * 0.01, end_time * 0.01, steps=steps + 1).float().to(device)

                    print('input frame %s of video %s ======>' % (key_frame_name,video_name))
                    print('generate %d frames' % (steps))

                    frame_pred, boxes_pred = netG(img, obj, triple, obj_to_img=None, boxes_gt=bbox_gt,
                                                                 masks_gt=None,
                                                                 time_stamps=time_stamps)
                    print('finish')

                    frame_fake=frame_pred.detach()
                    frame_real = dataset_train.get_real_frame(start, end, video_name).to(device)

                    # if batch_idx % 200 == 0:
                    if video_name=='001YG.mp4':
                        print('Saving Video%d in epoch %d %s generated frames' % (batch_idx, epoch, video_name))

                        imgs_pred = imagenet_deprocess_batch(frame_pred)
                        imgs_real = imagenet_deprocess_batch(frame_real)

                        isExists = os.path.exists(args.generate_frame_path + video_name + '_epoch%d' % epoch)
                        if not isExists:
                            os.makedirs(args.generate_frame_path + video_name + '_epoch%d' % epoch)

                        with open(args.generate_frame_path + video_name + '_epoch%d' % epoch + '/triples.txt',
                                  'w') as f:
                            f.write(str(triple))

                        for j, img_pred in enumerate(imgs_pred):
                            img_pred = img_pred.numpy().transpose(1, 2, 0)

                            if j == 0:
                                imwrite(
                                    args.generate_frame_path + video_name + '_epoch%d' % epoch + '/%d_pred_KF.png' % save_name_pred,
                                    img_pred)
                            else:
                                imwrite(
                                    args.generate_frame_path + video_name + '_epoch%d' % epoch + '/%d_pred.png' % save_name_pred,
                                    img_pred)
                            save_name_pred += 1

                        for j, img_real in enumerate(imgs_real):
                            img_real = img_real.numpy().transpose(1, 2, 0)

                            imwrite(
                                args.generate_frame_path + video_name + '_epoch%d' % epoch + '/%d_real.png' % save_name_real,
                                img_real)
                            save_name_real += 1

                        print('========Save Finish========')

                    step+=1

                    loss_netG1 = compute_losses(args, boxes_pred, bbox_gt, frame_pred, frame_real)
                    output1 = netD(frame_pred).reshape(-1)
                    loss_netG2 = criterion(output1, torch.ones_like(output1))
                    loss_netG = loss_netG1 + loss_netG2

                    optimizer_netG.zero_grad()
                    loss_netG.backward()
                    optimizer_netG.step()

                    disc_real = netD(frame_real).reshape(-1)  # (N,1,1,1)→(N,)
                    loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
                    disc_fake = netD(frame_pred.detach()).reshape(-1)
                    loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                    loss_netD = (loss_disc_real + loss_disc_fake) / 2

                    print('======loss_netG:%f======' % loss_netG)
                    print('======loss_netD:%f======' % loss_netD)

                    if i % 5 == 0:
                        optimizer_netD.zero_grad()
                        loss_netD.backward()
                        optimizer_netD.step()

                    G_losses.append(loss_netG.item())
                    D_losses.append(loss_netD.item())
                    step += 1

                    writer.add_scalar('LossG per KF', loss_netG, global_step=step)
                    writer.add_scalar('LossD per KF', loss_netD, global_step=step)

            if epoch % 1 == 0:
                checkpoint = {'netG_state_dict': netG.state_dict(), 'netD_state_dict': netD.state_dict(),
                              'optimizer_netG': optimizer_netG.state_dict(), 'optimizer_netD': optimizer_netD.state_dict()}

                save_checkpoint(checkpoint, args, epoch)

    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    savefig(args.savefig_path + 'loss100_SG_epoch%d.jpg' % epoch)



if __name__ == '__main__':
    main(args)





