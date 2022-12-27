import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):

    def __init__(self, im_chan=3, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.InstanceNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):

        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)

class GlobalAvgPool(nn.Module):
  def forward(self, x):
    N, C = x.size(0), x.size(1)
    return x.view(N, C, -1).mean(dim=2)

def tensor_linspace(start, end, steps=10):

  assert start.size() == end.size()
  view_size = start.size() + (1,)
  w_size = (1,) * start.dim() + (steps,)
  out_size = start.size() + (steps,)

  start_w = torch.linspace(1, 0, steps=steps).to(start)
  start_w = start_w.view(w_size).expand(out_size)
  end_w = torch.linspace(0, 1, steps=steps).to(start)
  end_w = end_w.view(w_size).expand(out_size)

  start = start.contiguous().view(view_size).expand(out_size)
  end = end.contiguous().view(view_size).expand(out_size)

  out = start_w * start + end_w * end
  return out

class obj_Discriminator(nn.Module):
    def __init__(self,args,vocab):
        super(obj_Discriminator, self).__init__()

        self.vocab=vocab
        self.object_size=args.object_size
        self.args=args

        obj_disc_layers=[nn.Conv2d(3,64,kernel_size=4,stride=2),
                         nn.InstanceNorm2d(64),
                         nn.LeakyReLU(0.2),
                         nn.Conv2d(64,128,kernel_size=4,stride=2),
                         nn.InstanceNorm2d(128),
                         nn.LeakyReLU(0.2),
                         nn.Conv2d(128,256,kernel_size=4,stride=2),
                         ]

        nn.init.kaiming_normal_(obj_disc_layers[0].weight)
        nn.init.kaiming_normal_(obj_disc_layers[3].weight)
        nn.init.kaiming_normal_(obj_disc_layers[6].weight)

        self.cnn=nn.Sequential(*obj_disc_layers)

        self.obj_disc=nn.Sequential(self.cnn,GlobalAvgPool(),nn.Linear(256,512))
        num_objects=len(vocab['object_classes_itn'])
        self.real_classifier=nn.Sequential(nn.Linear(512,1),nn.Sigmoid())
        self.obj_classifier=nn.Linear(512,num_objects)

    def forward(self,img,objs,boxes):
        N,C,H,W=img.size()
        HH,WW=self.object_size,self.object_size
        n=len(objs)
        feats=img.view(1,C,H,W).expand(n,C,H,W).contiguous()
        bbox = 2 * boxes - 1
        x0, y0 = bbox[:, 0], bbox[:, 1]
        x1, y1 = bbox[:, 2], bbox[:, 3]
        X = tensor_linspace(x0, x1, steps=WW).view(n, 1, WW).expand(n, HH, WW)
        Y = tensor_linspace(y0, y1, steps=HH).view(n, HH, 1).expand(n, HH, WW)
        grid = torch.stack([X, Y], dim=3)
        crops=F.grid_sample(feats, grid)

        if crops.dim() == 3:
            crops = crops[:, None]

        vecs = self.obj_disc(crops)
        real_scores = self.real_classifier(vecs)
        obj_scores = self.obj_classifier(vecs)
        ac_loss = F.cross_entropy(obj_scores, objs)*self.args.ac_loss_weight
        return real_scores, ac_loss