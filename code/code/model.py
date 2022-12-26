import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint


#GCN
class GraphTripleConv(nn.Module):
    def __init__(self, input_dim, output_dim=None, hidden_dim=256):
        super(GraphTripleConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        net1_layers=[nn.Linear(input_dim*3,hidden_dim,bias=True),
                     nn.ReLU(),
                     nn.Linear(hidden_dim,2*hidden_dim+output_dim,bias=True),
                     nn.ReLU(),
        ]
        nn.init.kaiming_normal_(net1_layers[0].weight)
        nn.init.kaiming_normal_(net1_layers[2].weight)
        self.net1=nn.Sequential(*net1_layers)

        net2_layers = [nn.Linear(hidden_dim, hidden_dim, bias=True),
                       nn.ReLU(),
                       nn.Linear(hidden_dim, output_dim,bias=True),
                       nn.ReLU(),
                       ]
        nn.init.kaiming_normal_(net2_layers[0].weight)
        nn.init.kaiming_normal_(net2_layers[2].weight)
        self.net2 = nn.Sequential(*net2_layers)


    def forward(self, obj_vecs, pred_vecs, edges):

        dtype, device = obj_vecs.dtype, obj_vecs.device
        O, T = obj_vecs.size(0), pred_vecs.size(0)
        Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        # Get current vectors for subjects and objects; these have shape (T, Din)
        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]

        # Get current vectors for triples; shape is (T, 3 * Din)
        # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        new_t_vecs = self.net1(cur_t_vecs)

        # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
        # p vecs have shape (T, Dout)
        new_s_vecs = new_t_vecs[:, :H]
        new_p_vecs = new_t_vecs[:, H:(H + Dout)]
        new_o_vecs = new_t_vecs[:, (H + Dout):(2 * H + Dout)]

        # Allocate space for pooled object vectors of shape (O, H)
        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

        # Use scatter_add to sum vectors for objects that appear in multiple triples;
        # we first need to expand the indices to have shape (T, D)
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        obj_counts = torch.zeros(O, dtype=dtype, device=device)
        ones = torch.ones(T, dtype=dtype, device=device)
        obj_counts = obj_counts.scatter_add(0, s_idx, ones)
        obj_counts = obj_counts.scatter_add(0, o_idx, ones)

        obj_counts = obj_counts.clamp(min=1)
        pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

        # Send pooled object vectors through net2 to get output object vectors,
        # of shape (O, Dout)
        new_obj_vecs = self.net2(pooled_obj_vecs)

        return new_obj_vecs, new_p_vecs

class GraphTripleConvNet(nn.Module):

    def __init__(self, input_dim, num_layers=5, hidden_dim=256):
        super(GraphTripleConvNet, self).__init__()
        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim
        }
        for _ in range(self.num_layers):
            self.gconvs.append(GraphTripleConv(**gconv_kwargs))

    def forward(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs

def boxes_to_layout(vecs, boxes, obj_to_img, H, W=None, pooling='sum'):
    O, D = vecs.size()
    if W is None:
        W = H
    grid = _boxes_to_grid(boxes, H, W)
    img_in = vecs.view(O, D, 1, 1).expand(O, D, 8, 8)
    sampled = F.grid_sample(img_in, grid)
    out = _pool_samples(sampled, obj_to_img, pooling=pooling)

    return out

def masks_to_layout(vecs, boxes, masks, obj_to_img, H, W=None, pooling='sum'):
    O, D = vecs.size()
    M = masks.size(1)
    assert masks.size() == (O, M, M)
    if W is None:
        W = H
    grid = _boxes_to_grid(boxes, H, W)
    img_in = vecs.view(O, D, 1, 1) * masks.float().view(O, 1, M, M)
    sampled = F.grid_sample(img_in, grid)
    out = _pool_samples(sampled, obj_to_img, pooling=pooling)
    return out

def _boxes_to_grid(boxes, H, W):
    O = boxes.size(0)
    boxes = boxes.view(O, 4, 1, 1)

    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = boxes[:, 2], boxes[:, 3]
    ww = x1 - x0
    hh = y1 - y0
    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)
    grid = grid.mul(2).sub(1)

    return grid

def _pool_samples(samples, obj_to_img, pooling='sum'):
    dtype, device = samples.dtype, samples.device
    O, D, H, W = samples.size()
    N = obj_to_img.data.max().item() + 1

    out = torch.zeros(N, D, H, W, dtype=dtype, device=device)
    idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)
    out = out.scatter_add(0, idx, samples)

    if pooling == 'avg':
        ones = torch.ones(O, dtype=dtype, device=device)
        obj_counts = torch.zeros(N, dtype=dtype, device=device)
        obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
        print(obj_counts)
        obj_counts = obj_counts.clamp(min=1)
        out = out / obj_counts.view(N, 1, 1, 1)
    elif pooling != 'sum':
        raise ValueError('Invalid pooling "%s"' % pooling)

    return out


#RCN
class RefinementModule(nn.Module):
    def __init__(self, layout_dim, input_dim, output_dim):
        super(RefinementModule, self).__init__()
        layers = [nn.Conv2d(layout_dim+input_dim,output_dim,kernel_size=3,padding=1),
                  nn.InstanceNorm2d(output_dim),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(output_dim,output_dim,kernel_size=3,padding=1),
                  nn.InstanceNorm2d(output_dim),
                  ]

        nn.init.kaiming_normal_(layers[0].weight)
        nn.init.kaiming_normal_(layers[3].weight)

        self.net = nn.Sequential(*layers)

    def forward(self, layout, feats):
        _, _, HH, WW = layout.size()
        _, _, H, W = feats.size()
        assert HH >= H
        if HH > H:
            factor = round(HH // H)
            assert HH % factor == 0
            assert WW % factor == 0 and WW // factor == W
            layout = F.avg_pool2d(layout, kernel_size=factor, stride=factor)
        net_input = torch.cat([layout, feats], dim=1)
        out = self.net(net_input)
        return out

class RefinementNetwork(nn.Module):
    def __init__(self, dims):
        super(RefinementNetwork, self).__init__()
        layout_dim = dims[0]
        self.refinement_modules = nn.ModuleList()
        for i in range(1, len(dims)):
            input_dim = 1 if i == 1 else dims[i - 1]
            output_dim = dims[i]
            mod = RefinementModule(layout_dim, input_dim, output_dim)
            self.refinement_modules.append(mod)

        output_conv_layers = [
            nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
            nn.InstanceNorm2d(dims[-1]),
            nn.Conv2d(dims[-1], 3, kernel_size=1, padding=0)
        ]
        nn.init.kaiming_normal_(output_conv_layers[0].weight)
        nn.init.kaiming_normal_(output_conv_layers[2].weight)
        self.output_conv = nn.Sequential(*output_conv_layers)

    def forward(self, layout):
        N, _, H, W = layout.size()
        self.layout = layout

        input_H, input_W = H, W
        for _ in range(len(self.refinement_modules)):
            input_H //= 2
            input_W //= 2

        assert input_H != 0
        assert input_W != 0

        feats = torch.zeros(N, 1, input_H, input_W).to(layout)
        for mod in self.refinement_modules:
            feats = F.upsample(feats, scale_factor=2, mode='nearest')
            feats = mod(layout, feats)

        out = self.output_conv(feats)
        return out


#Features Extracting
class Sg2ImModel(nn.Module):
    def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
                 gconv_dim=128, gconv_hidden_dim=512,
                 gconv_num_layers=5,
                 mask_size=None):

        super(Sg2ImModel, self).__init__()

        self.vocab = vocab
        self.image_size = image_size

        num_objs = len(vocab['object_classes_itn'])
        num_relation = len(vocab['relationship_classes_itn'])

        self.obj_embeddings = nn.Embedding(num_objs, embedding_dim)
        self.pred_embeddings = nn.Embedding(num_relation, embedding_dim)

        gconv_kwargs = {
            'input_dim': embedding_dim,
            'output_dim': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
        }
        self.gconv = GraphTripleConv(**gconv_kwargs)

        gconv_kwargs = {
            'input_dim': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'num_layers': gconv_num_layers - 1,
        }
        self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        box_net_layers = [nn.Linear(gconv_dim, gconv_hidden_dim, bias=True),
                          nn.ReLU(),
                          nn.Linear(gconv_hidden_dim, 4, bias=True),
                          nn.ReLU(),
                       ]
        nn.init.kaiming_normal_(box_net_layers[0].weight)
        nn.init.kaiming_normal_(box_net_layers[2].weight)
        self.box_net = nn.Sequential(*box_net_layers)
        self.mask_net = self._build_mask_net(gconv_dim, mask_size)

        image_layers = [nn.Conv2d(3, 32, 3, 1, 1),
                        nn.InstanceNorm2d(32),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(32, 64, 3, 1, 1)]

        nn.init.kaiming_normal_(image_layers[0].weight)
        nn.init.kaiming_normal_(image_layers[3].weight)
        self.image_net = nn.Sequential(*image_layers)

    def _build_mask_net(self,dim, mask_size):
        output_dim = 1
        layers, cur_size = [], 1
        while cur_size < mask_size:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.BatchNorm2d(dim))
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            cur_size *= 2
        if cur_size != mask_size:
            raise ValueError('Mask size must be a power of 2')
        layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self,img, objs, triples, obj_to_img=None,
                boxes_gt=None, masks_gt=None):

        O, T = objs.size(0), triples.size(0)
        if len(triples.size())==1:
            s=triples[0]
            p=triples[1]
            o=triples[2]
            edges=torch.stack([s,o])
        else:
            s, p, o = triples.chunk(3, dim=1)
            s, p, o = [x.squeeze(1) for x in [s, p, o]]
            edges = torch.stack([s, o], dim=1)

        if obj_to_img is None:
            obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

        obj_vecs = self.obj_embeddings(objs)
        pred_vecs = self.pred_embeddings(p)

        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        boxes_pred = self.box_net(obj_vecs)

        mask_scores = self.mask_net(obj_vecs.view(O, -1, 1, 1))
        masks_pred = mask_scores.squeeze(1).sigmoid()

        H, W = self.image_size
        layout_boxes = boxes_pred if boxes_gt is None else boxes_gt  # (1,O,4) boxes_gt is not None

        layout_masks = masks_pred if masks_gt is None else masks_gt
        layout_features = masks_to_layout(obj_vecs, layout_boxes, layout_masks,obj_to_img, H, W)  # (batch_size(1),128,64,64)

        image_features = self.image_net(img)
        features = torch.cat((layout_features, image_features), dim=1)

        return features,boxes_pred


#ODE
class ODEfunc(nn.Module):

    def __init__(self, dim1,dim2):
        super(ODEfunc, self).__init__()
        self.conv11 = nn.Conv2d(dim1, dim1*2, 3, 1, 1)
        self.conv12 = nn.Conv2d(dim1*2, dim1, 3, 1, 1)

        self.conv21 = nn.Conv2d(dim2, dim2*2, 3, 1, 1)
        self.conv22 = nn.Conv2d(dim2*2, dim2, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        x1=x[:,0:128,:,:]
        x2=x[:,128:,:,:]

        out1 = self.conv11(x1)
        out1 = self.relu(out1)
        out1 = self.conv12(out1)
        out1 = self.relu(out1)

        out2 = self.conv21(x2)
        out2 = self.relu(out2)
        out2 = self.conv22(out2)
        out2 = self.relu(out2)

        out=torch.cat((out1,out2),dim=1)
        return out


#Features to Image
class sgODE_model(nn.Module):

    def __init__(self, args, vocab, device):
        super(sgODE_model, self).__init__()
        self.args = args
        self.vocab = vocab
        self.device = device

        sg_kwargs = {
            'vocab': self.vocab,
            'image_size': self.args.image_size,
            'embedding_dim': self.args.embedding_dim,
            'gconv_dim': self.args.gconv_dim,
            'gconv_hidden_dim': self.args.gconv_hidden_dim,
            'gconv_num_layers': self.args.gconv_num_layers,
            'mask_size': self.args.mask_size
        }

        self.feature_model = Sg2ImModel(**sg_kwargs).to(self.device)

        self.odefunc = ODEfunc(dim1=128,dim2=64).to(self.device)

        refinement_dims = (1024, 512, 256, 128, 64)
        refinement_kwargs = {
            'dims': (64+128 + self.args.layout_noise_dim,) + refinement_dims}

        self.refinement_net = RefinementNetwork(**refinement_kwargs).to(self.device)

    def forward(self, img, objs, triples, obj_to_img=None,
                       boxes_gt=None, masks_gt=None, time_stamps=None):

        features, boxes_pred = self.feature_model(img, objs, triples, obj_to_img, boxes_gt, masks_gt)

        self.boxes_gt = boxes_gt

        self.time_stamps = time_stamps.type_as(features)
        features_new = odeint(self.odefunc, features, self.time_stamps, rtol=self.args.tol, atol=self.args.tol)
        features_new = features_new.squeeze()

        layout=features_new[:,0:128,:,:]
        image_features=features_new[:,128:,:,:]

        self.layout_noise_dim = self.args.layout_noise_dim
        N, C, H, W = layout.size()
        noise_shape = (N, self.layout_noise_dim, H, W)
        layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
                                   device=layout.device)
        layout = torch.cat([image_features,layout,layout_noise], dim=1)
        frame_pred = self.refinement_net(layout)  # (num,3,64,64)

        return frame_pred, boxes_pred















