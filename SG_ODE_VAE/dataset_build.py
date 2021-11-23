import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from utils import imagenet_preprocess,Resize

import numpy as np
import PIL



class AGDataset(Dataset):
  def __init__(self, vocab, kf_info,video_info,frames_path, image_size=(64, 64),
               normalize_images=True):

    super(AGDataset, self).__init__()

    self.vocab = vocab
    self.kf_info=kf_info
    self.video_info=video_info
    self.frames_path=frames_path
    self.image_size = image_size
    self.num_objects = len(vocab['object_classes_itn'])

    transform = [Resize(image_size), T.ToTensor()]
    if normalize_images:
      transform.append(imagenet_preprocess())
    self.transform = T.Compose(transform)

  def __len__(self):
    num = len(self.kf_info.keys())
    return num

  def __getitem__(self, index):

    for i,(k,v) in enumerate(self.kf_info.items()):
      if i!=index:
        continue
      else:

        with open(self.frames_path+k,'rb') as f:
          with PIL.Image.open(f) as image:
            WW,HH=image.size
            key_frame=self.transform(image.convert('RGB'))


        key_frame_name=k

        H,W=self.image_size

        O=len(v['obj'])+1
        objs = torch.LongTensor(O).fill_(-1)

        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
        obj_idx_mapping = {}

        for i, obj in enumerate(v['obj']):
          objs[i]=obj
          x0, y0, x1, y1 = v['bbox'][i]
          x0 = float(x0) / WW
          y0 = float(y0) / HH
          x1 = float(x1) / WW
          y1 = float(y1) / HH
          boxes[i] = torch.FloatTensor([x0, y0, x1, y1])#the last boxes is 0 0 1 1
          obj_idx_mapping[obj] = i

        # The last object will be the special __image__ object
        objs[O-1]=self.vocab['object_classes_nti']['__image__']

        triples = []
        for j,r in enumerate(v['relationship']):
          if r==0:
            continue
          s = 1
          p = r
          o = v['obj'][j+1]
          s = obj_idx_mapping.get(s, None)
          o = obj_idx_mapping.get(o, None)
          if s is not None and o is not None:
            triples.append([s, p, o])

        # Add dummy __in_image__ relationships for all objects
        in_image = self.vocab['relationship_classes_nti']['__in_image__']
        for i in range(O - 1):
          triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)
        break

    return key_frame_name,key_frame,objs,triples,boxes

  def get_video_info(self,index):

    for i,(k,v) in enumerate(self.video_info.items()):
      if i!=index:
        continue
      else:
        video_name=k
        frames_num=len(v)
        frame_name=[]
        key_frame_inter = []

        for num, info in enumerate(v):  # info of each key_frame
          frame_name.append(info['id'])

        for i in range(0,frames_num-1):
          # if i==frames_num-1:
          #   key_frame_inter.append([int(frame_name[i].split('.')[0].lstrip('0')),int(frame_name[i].split('.')[0].lstrip('0'))+10])
          # else:
          key_frame_inter.append([int(frame_name[i].split('.')[0].lstrip('0')),int(frame_name[i+1].split('.')[0].lstrip('0'))])
        break

    return video_name,frames_num,key_frame_inter


  def get_real_frame(self,start,end,video_name):

    frame_real_list=[]
    n=end-start
    if n>10:
      n=10

    for i in np.linspace(start,end,n+1):
      with open(self.frames_path + video_name + '/%.6d'%int(i)+'.png', 'rb') as f:
        with PIL.Image.open(f) as image:
          image = self.transform(image.convert('RGB'))
          image=image.unsqueeze(dim=0)
      frame_real_list.append(image)
    frame_real=torch.cat(frame_real_list,dim=0)

    return frame_real







