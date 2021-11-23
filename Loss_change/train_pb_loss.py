import os
import argparse

from imageio import imwrite

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from annotations_700 import build_vocab, get_info, get_700_info
from dataset_build import AGDataset
from model_pb import sgODE_model, Discriminator, compute_losses
from utils import imagenet_deprocess_batch

parser = argparse.ArgumentParser()

# path
# parser.add_argument('--frames_path', default='F:/dataset/Action Genome/frames/')
# parser.add_argument('--annotations_path', default='./dataset/annotations/')
# parser.add_argument('--csv_path',default='./dataset/vocab_kf_info/')
parser.add_argument('--frames_path', default='/data/scene_understanding/action_genome/frames/')
parser.add_argument('--annotations_path', default='/data/scene_understanding/action_genome/annotations/')
parser.add_argument('--checkpoint_path', default='./')
parser.add_argument('--generate_frame_path', default='./generation_700_pb/')

# neural network set
parser.add_argument('--image_size', default=(64, 64))
parser.add_argument('--nepoch', default=10)
# parser.add_argument('--video_num_iterations',default=5)#number of the video for iterations
parser.add_argument('--lr_G', type=float, default=1e-5)
parser.add_argument('--lr_D', type=float, default=1e-7)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--mask_size', default=16, type=int)  # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--refinement_network_dims', default=(1024, 512, 256, 128, 64))
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--loader_num_workers', default=0, type=int)
# parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

# ODE set
parser.add_argument('--tol', type=float, default=1e-3)  # tolerance
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--time_step', type=float, default=0.01)

# netG Loss set
# parser.add_argument('--relationship_loss_weight', default=1.0, type=float)
parser.add_argument('--bbox_loss_weight', default=10.0, type=float)
parser.add_argument('--img_loss_weight', default=10.0, type=float)
# parser.add_argument('--iimg_loss_weight', default=1.0, type=float)
parser.add_argument('--pb_loss_weight', default=5.0, type=float)

# other set
parser.add_argument('--nonperson_filer', default='True')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--checkpoint_num', default=700, type=int)
parser.add_argument('--load_model', default=False)
parser.add_argument('--savefig_path', default='./')

args = parser.parse_args()


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def save_checkpoint(state, args, epoch):
    print('Saving checkpoint in epoch %d' % epoch)
    file = args.checkpoint_path + 'my_checkpoint_700_%d.pth.tar' % epoch
    torch.save(state, file)


def load_checkpoint(checkpoint, netG, netD, optimizer_netG, optimizer_netD):
    print('Loading checkpoint')
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizer_netG.load_state_dict(checkpoint['optimizer_netG'])
    optimizer_netD.load_state_dict(checkpoint['optimizer_netD'])


def main(args):
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # device='cpu'

    # build vocab and key_frames_informations
    vocab = build_vocab(args.annotations_path)
    kf_info, video_info, train_list = get_info(args.annotations_path, args.nonperson_filer, vocab)
    kf_info_700, video_info_700 = get_700_info(kf_info, video_info)

    dset_kwargs = {
        'vocab': vocab,
        'kf_info': kf_info_700,
        'video_info': video_info_700,
        'frames_path': args.frames_path,
        'image_size': args.image_size,
        'normalize_images': True
    }
    dataset = AGDataset(**dset_kwargs)

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': False,
    }

    loader = DataLoader(dataset, **loader_kwargs)
    data_gen = inf_generator(loader)

    netG = sgODE_model(args, vocab, device)
    netD = Discriminator(input_channel=3, hidden=64).to(device)

    optimizer_netG = optim.Adam(netG.parameters(), lr=args.lr_G, betas=(0.5, 0.999))
    optimizer_netD = optim.Adam(netD.parameters(), lr=args.lr_D, betas=(0.5, 0.999))

    if args.load_model:
        load_checkpoint(torch.load(args.checkpoint_path + 'my_checkpoint_700_1.pth.tar'), netG, netD, optimizer_netG,
                        optimizer_netD)

    criterion = nn.BCELoss().to(device)
    writer = SummaryWriter(f'./runs/out_700_pb')

    netG.train()
    netD.train()

    step = 0
    train_num = 0

    G_losses = []
    D_losses = []

    G_losses_avg = []
    D_losses_avg = []

    G_losses_avg1 = []
    D_losses_avg1 = []

    # for epoch in range(0,args.nepoch-5):
    for epoch in range(2, 50):

        # if epoch % 1 == 0:
        #     checkpoint = {'netG_state_dict': netG.state_dict(), 'netD_state_dict': netD.state_dict(),
        #                   'optimizer_netG': optimizer_netG.state_dict(), 'optimizer_netD': optimizer_netD.state_dict()}
        #
        #     save_checkpoint(checkpoint, args, epoch)

        # for video in range(0,len(video_info)):
        for video in range(0, 700):
            print('generate video Nummber %d' % video)

            # if video>=args.video_num_iterations:
            #     break
            video_name, frames_num, key_frame_inter = dataset.get_video_info(video)

            # if video_name in train_list:#(train_list 7502) train_num 7421
            #     train_num+=1

            save_name_pred = 0
            # save_name_ipred=0
            save_name_real = 0

            # if train_num % args.checkpoint_num == 0:
            #     checkpoint = {'netG_state_dict': netG.state_dict(), 'netD_state_dict': netD.state_dict(),
            #                   'optimizer_netG': optimizer_netG.state_dict(), 'optimizer_netD': optimizer_netD.state_dict()}
            #     save_checkpoint(checkpoint, args,epoch)
            #
            # print('Starting video %s Generation Video Nummber %d' % (video_name,train_num))

            for i in range(0, frames_num):

                key_frame_name, key_frame, objs, triples, boxes = data_gen.__next__()

                # only the video in train_list
                # if video_name not in train_list:
                #     continue

                if i == frames_num - 1:  # not using the last Key Frame, because the frame num after the last KF is not sure
                    continue

                else:

                    objs = objs.squeeze().to(device)
                    triples = triples.squeeze().to(device)
                    boxes = boxes.squeeze().to(device)

                    inter = key_frame_inter[i]

                    # Using the key frame as input
                    # img=key_frame.to(device)

                    # Using the last frame as input
                    if i == 0:
                        img = key_frame.to(device)  # (1,3,64,64)
                    else:
                        img = frame_last.unsqueeze(dim=0)  # (1,3,64,64)

                    video_start_time = int(inter[0])
                    start = int(inter[0])
                    end = int(inter[1])

                    start_time = start - video_start_time
                    end_time = end - video_start_time

                    steps = (end_time - start_time) if (end_time - start_time) < 10 else 10

                    time_stamps = torch.linspace(start_time * 0.01, end_time * 0.01, steps=steps + 1).float().to(device)

                    print('input frame %s ======>' % key_frame_name)
                    print('generate %d frames of video %s' % (steps, video_name))

                    # frame_pred,boxes_pred,frame_pred_from_image=netG.frame_generate(img,objs, triples, obj_to_img=None,boxes_gt=boxes,
                    #                                           masks_gt=None,time_stamps=time_stamps)

                    frame_pred, boxes_pred = netG.frame_generate(img, objs, triples, obj_to_img=None, boxes_gt=boxes,
                                                                 masks_gt=None,
                                                                 time_stamps=time_stamps)

                    frame_last = frame_pred[-1].detach()
                    # frame_last=frame_pred[-1].clone()

                    print('===Generation Finish===')

                    frame_real = dataset.get_real_frame(start, end, video_name).to(device)

                    if video % 200 == 0:
                        print('Saving Video%d in epoch %d %s generated frames' % (video, epoch, video_name))

                        imgs_pred = imagenet_deprocess_batch(frame_pred)
                        imgs_real = imagenet_deprocess_batch(frame_real)

                        isExists = os.path.exists(args.generate_frame_path + video_name + '_epoch%d' % epoch)
                        if not isExists:
                            os.makedirs(args.generate_frame_path + video_name + '_epoch%d' % epoch)

                        with open(args.generate_frame_path + video_name + '_epoch%d' % epoch + '/triples.txt',
                                  'w') as f:
                            f.write(str(triples))

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

                # frame_pred.detach_()
                # frame_pred_from_image.detach_()
                # frame(frame_num,3,64,64)

                loss_netG1 = compute_losses(args, boxes_pred, boxes, frame_pred, frame_real)
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

                if step % 100 == 0:  # average

                    LossG = sum(G_losses[-100:]) / 100
                    LossD = sum(D_losses[-100:]) / 100

                    G_losses_avg.append(LossG)
                    D_losses_avg.append(LossD)

                    writer.add_scalar('LossG per 100 KF', LossG, global_step=step / 100)
                    writer.add_scalar('LossD per 100 KF', LossD, global_step=step / 100)

                if step % 1000 == 0:  # average

                    LossG1 = sum(G_losses[-1000:]) / 1000
                    LossD1 = sum(D_losses[-1000:]) / 1000

                    G_losses_avg1.append(LossG1)
                    D_losses_avg1.append(LossD1)

        if epoch % 1 == 0:
            checkpoint = {'netG_state_dict': netG.state_dict(), 'netD_state_dict': netD.state_dict(),
                          'optimizer_netG': optimizer_netG.state_dict(), 'optimizer_netD': optimizer_netD.state_dict()}

            save_checkpoint(checkpoint, args, epoch)

    # plt.title("Generator and Discriminator Loss During Training ")
    # x1=range(len(G_losses))
    # plt.plot(x1,G_losses,label='G')
    # plt.plot(x1,D_losses, label="D")
    # plt.xlabel("Key_Frames_Num")
    # plt.ylabel("Loss")
    # plt.legend()
    # savefig(args.savefig_path + 'loss_dataset700_epoch%d.jpg' % epoch)

    plt.title("Generator and Discriminator Loss During Training per 100 steps")
    x2 = range(len(G_losses_avg))
    plt.plot(x2, G_losses_avg, label="G")
    plt.plot(x2, D_losses_avg, label="D")
    plt.xlabel("Key_Frames_Num")
    plt.ylabel("Loss")
    plt.legend()
    savefig(args.savefig_path + 'loss100_dataset700_epoch%d.jpg' % epoch)

    # plt.title("Generator and Discriminator Loss During Training per 1000 steps")
    # x3 = range(len(G_losses_avg1))
    # plt.plot(x3, G_losses_avg1, label="G")
    # plt.plot(x3, D_losses_avg1, label="D")
    # plt.xlabel("Key_Frames_Num")
    # plt.ylabel("Loss")
    # plt.legend()
    # savefig(args.savefig_path + 'loss1000_dataset700_epoch%d.jpg' % epoch)


if __name__ == '__main__':
    main(args)





