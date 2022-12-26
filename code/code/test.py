import os
import argparse
from imageio import imwrite

import torch
from torch.utils.data import DataLoader

from annotations import build_vocab,get_info
from dataset_build import AGDataset_test
from model import sgODE_model
from utils import imagenet_deprocess_batch

parser = argparse.ArgumentParser()

#path
parser.add_argument('--frames_path', default='/data/scene_understanding/action_genome/frames/')
parser.add_argument('--annotations_path', default='/data/scene_understanding/action_genome/annotations/')
parser.add_argument('--checkpoint_path',default='./')
parser.add_argument('--generate_frame_path',default='./test/')

#neural network set
parser.add_argument('--image_size', default=(64,64))
parser.add_argument('--nepoch',default=20)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--mask_size', default=16, type=int)
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--refinement_network_dims', default=(1024,512,256,128,64))
parser.add_argument('--normalization', default='batch')
parser.add_argument('--layout_noise_dim', default=64, type=int)
parser.add_argument('--loader_num_workers', default=0, type=int)

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


def run(args):

    vocab = build_vocab(args.annotations_path)
    kf_info, video_info, train_list, test_list, max_length = get_info(args.annotations_path, args.nonperson_filer,
                                                                      vocab)

    dset_test_kwargs = {
        'vocab': vocab,
        'video_info': video_info,
        'frames_path': args.frames_path,
        'image_size': args.image_size,
        'test_list': test_list,
    }
    dataset_test = AGDataset_test(**dset_test_kwargs)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)

    netG = sgODE_model(args, vocab, device).to(device)

    checkpoint=torch.load(args.checkpoint_path+'my_checkpoint_SG.pth.tar')
    netG.load_state_dict(checkpoint['netG_state_dict'])

    netG.eval()

    for batch_idx, (video_name, frames_num, key_frame, frame_name, key_frame_inter, obj_list, triple_list,
                    bbox_list) in enumerate(test_loader):

        video_name = video_name[0]
        frames_num = int(frames_num)

        save_name_pred = 0
        save_name_real=0

        for i in range(0, frames_num):

            if i == frames_num - 1:
                continue

            else:

                obj = obj_list[i].squeeze(0).to(device)
                triple = triple_list[i].squeeze(0).to(device)
                bbox_gt = bbox_list[i].squeeze(0).to(device)

                inter = key_frame_inter[i]
                key_frame_name = frame_name[i][0]

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

                time_stamps = torch.linspace(start_time * 0.01, end_time * 0.01, steps=steps + 1).float().to(
                    device)

                print('input frame %s of video %s ======>' % (key_frame_name, video_name))

                frame_pred, boxes_pred = netG(img, obj, triple, obj_to_img=None, boxes_gt=bbox_gt,
                                              masks_gt=None,
                                              time_stamps=time_stamps)
                print('finish')

                frame_fake = frame_pred.detach()
                frame_real = dataset_test.get_real_frame(start, end, video_name).to(device)


                print('Saving Video%d %s generated frames' % (batch_idx, video_name))

                imgs_pred = imagenet_deprocess_batch(frame_pred)
                imgs_real = imagenet_deprocess_batch(frame_real)

                isExists = os.path.exists(args.generate_frame_path + video_name )
                if not isExists:
                    os.makedirs(args.generate_frame_path + video_name )

                with open(args.generate_frame_path + video_name + '/triples.txt',
                          'w') as f:
                    f.write(str(triple))

                for j, img_pred in enumerate(imgs_pred):
                    img_pred = img_pred.numpy().transpose(1, 2, 0)

                    if j == 0:
                        imwrite(
                            args.generate_frame_path + video_name  + '/%d_pred_KF.png' % save_name_pred,
                            img_pred)
                    else:
                        imwrite(
                            args.generate_frame_path + video_name  + '/%d_pred.png' % save_name_pred,
                            img_pred)
                    save_name_pred += 1

                for j, img_real in enumerate(imgs_real):
                    img_real = img_real.numpy().transpose(1, 2, 0)

                    imwrite(
                        args.generate_frame_path + video_name + '/%d_real.png' % save_name_real,
                        img_real)
                    save_name_real += 1

                print('========Save Finish========')


if __name__ == '__main__':
    run(args)