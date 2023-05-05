import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
import cv2



class AGDataset_train(Dataset):
  def __init__(self, vocab, video_info,frames_path, image_size=(64, 64),train_list=None):

    super(AGDataset_train, self).__init__()

    self.vocab = vocab
    self.video_info=video_info
    self.frames_path=frames_path
    self.image_size = image_size
    self.train_list=train_list
    self.num_objects = len(vocab['object_classes_itn'])

    self.transform = T.Compose([
    T.Resize(image_size),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

  def __len__(self):
    num = len(self.train_list)
    return num

  def __getitem__(self, index):
    video_name=self.train_list[index]
    frames_num=len(self.video_info[video_name])
    frame_name = [] # str
    key_frame = []  # img(3,64,64) vs pred_image
    key_frame_inter = []  # time
    bbox_list=[]
    obj_list=[]
    triple_list=[]

    for num,info in enumerate(self.video_info[video_name]):

      name=info['id']

      frame_name.append(name)

      img=cv2.imread(self.frames_path+video_name+'/'+name)
      WW=img.shape[1]
      HH=img.shape[0]
      image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      image = PIL.Image.fromarray(image)
      frame =self.transform(image)
      key_frame.append(frame)

      O=len(info['obj'])+1
      objs=torch.LongTensor(O).fill_(-1)

      boxes= torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
      obj_idx_mapping={}

      for i,obj in enumerate(info['obj']):
        objs[i]=obj
        x0, y0, x1, y1 = info['bbox'][i]
        x0 = float(x0) / WW
        y0 = float(y0) / HH
        x1 = float(x1) / WW
        y1 = float(y1) / HH
        boxes[i]=torch.FloatTensor([x0, y0, x1, y1])
        obj_idx_mapping[obj]=i

      # The last object will be the special __image__ object
      objs[O - 1] = self.vocab['object_classes_nti']['__image__']

      triples = []
      for j, r in enumerate(info['relationship']):
        if r == 0:
          continue
        s = 1
        p = r
        o = info['obj'][j + 1]
        s = obj_idx_mapping.get(s, None)
        o = obj_idx_mapping.get(o, None)
        if s is not None and o is not None:
          triples.append([s, p, o])

      # Add dummy __in_image__ relationships for all objects
      in_image = self.vocab['relationship_classes_nti']['__in_image__']
      for i in range(O - 1):
        triples.append([i, in_image, O - 1])

      triples = torch.LongTensor(triples)

      obj_list.append(objs)
      triple_list.append(triples)
      bbox_list.append(boxes)

    for i in range(0,frames_num-1):
      key_frame_inter.append([int(frame_name[i].split('.')[0].lstrip('0')),int(frame_name[i+1].split('.')[0].lstrip('0'))])

    return video_name,frames_num,key_frame,frame_name,key_frame_inter,obj_list,triple_list,bbox_list

  def get_real_frame(self, start, end, video_name):

    frame_real_list = []
    n = end - start
    if n > 10:
      n = 10

    for i in np.linspace(start, end, n + 1):
      img = cv2.imread(self.frames_path + video_name + '/%.6d' % int(i) + '.png')
      image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      image = PIL.Image.fromarray(image)
      image = self.transform(image)
      image=image.unsqueeze(dim=0)
      frame_real_list.append(image)
      frame_real=torch.cat(frame_real_list,dim=0)

    # for i in np.linspace(start, end, n + 1):
    #   with open(self.frames_path + video_name + '/%.6d' % int(i) + '.png', 'rb') as f:
    #     with PIL.Image.open(f) as image:
    #       image = self.transform(image.convert('RGB'))
    #       image = image.unsqueeze(dim=0)
    #   frame_real_list.append(image)
    # frame_real = torch.cat(frame_real_list, dim=0)

    return frame_real


class AGDataset_test(Dataset):
  def __init__(self, vocab, video_info, frames_path, image_size=(64, 64), test_list=None):

    super(AGDataset_test, self).__init__()

    self.vocab = vocab
    self.video_info = video_info
    self.frames_path = frames_path
    self.image_size = image_size
    self.test_list = test_list
    self.num_objects = len(vocab['object_classes_itn'])

    self.transform = T.Compose([
      T.Resize(image_size),
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

  def __len__(self):
    num = len(self.test_list)
    return num

  def __getitem__(self, index):
    video_name = self.test_list[index]
    frames_num = len(self.video_info[video_name])
    frame_name = []  # str
    key_frame = []  # img(3,64,64) vs pred_image
    key_frame_inter = []  # time
    obj_list = []  # video obj of each key frame
    re_list = []  # video relationship of each key frame
    bbox_list = []

    for num, info in enumerate(self.video_info[video_name]):

      name = info['id']

      obj = info['obj']
      obj = torch.tensor(obj)
      relationship = info['relationship']
      relationship = torch.tensor(relationship)

      frame_name.append(name)
      obj_list.append(obj)
      re_list.append(relationship)

      img = cv2.imread(self.frames_path + video_name + '/' + name)
      WW = img.shape[1]
      HH = img.shape[0]
      image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      image = PIL.Image.fromarray(image)
      frame = self.transform(image)
      key_frame.append(frame)

      # with open(self.frames_path + video_name + '/' + name, 'rb') as f:
      #   with PIL.Image.open(f) as image:
      #     WW, HH = image.size
      #     frame = self.transform(image.convert('RGB'))
      #     key_frame.append(frame)

      box = info['bbox']
      box_list = []
      for b in box:
        x0, y0, x1, y1 = b
        x0 = float(x0) / WW
        y0 = float(y0) / HH
        x1 = float(x1) / WW
        y1 = float(y1) / HH
        bbox = torch.FloatTensor([x0, y0, x1, y1])
        box_list.append(bbox.unsqueeze(0))
      bbox_list.append(torch.cat(box_list, dim=0))

    for i in range(0, frames_num - 1):
      key_frame_inter.append(
        [int(frame_name[i].split('.')[0].lstrip('0')), int(frame_name[i + 1].split('.')[0].lstrip('0'))])

    return (video_name, frames_num, key_frame, frame_name, key_frame_inter, obj_list, re_list, bbox_list)


  def get_real_frame(self, start, end, video_name):

    frame_real_list = []
    n = end - start
    if n > 10:
      n = 10


    for i in np.linspace(start, end, n + 1):
      img = cv2.imread(self.frames_path + video_name + '/%.6d' % int(i) + '.png')
      image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      image = PIL.Image.fromarray(image)
      image = self.transform(image)
      image=image.unsqueeze(dim=0)
      frame_real_list.append(image)
      frame_real=torch.cat(frame_real_list,dim=0)

    # for i in np.linspace(start, end, n + 1):
    #   with open(self.frames_path + video_name + '/%.6d' % int(i) + '.png', 'rb') as f:
    #     with PIL.Image.open(f) as image:
    #       image = self.transform(image.convert('RGB'))
    #       image = image.unsqueeze(dim=0)
    #   frame_real_list.append(image)
    # frame_real = torch.cat(frame_real_list, dim=0)

    return frame_real
