import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
import argparse

parser = argparse.ArgumentParser()
# ODE set
parser.add_argument('--tol', type=float, default=1e-3)  # tolerance
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--time_step', type=float, default=0.01)

args = parser.parse_args()


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


def get_normalization_2d(channels, normalization):
    if normalization == 'instance':
        return nn.InstanceNorm2d(channels)
    elif normalization == 'batch':
        return nn.BatchNorm2d(channels)
    elif normalization=='group':
        return nn.GroupNorm(min(32,channels),channels)
    elif normalization == 'none':
        return None
    else:
        raise ValueError('Unrecognized normalization type "%s"' % normalization)


def get_activation(name):
    kwargs = {}
    if name.lower().startswith('leakyrelu'):
        if '-' in name:
            slope = float(name.split('-')[1])
            kwargs = {'negative_slope': slope}
    name = 'leakyrelu'
    activations = {
        'relu': nn.ReLU,
        'leakyrelu': nn.LeakyReLU,
    }
    if name.lower() not in activations:
        raise ValueError('Invalid activation "%s"' % name)
    return activations[name.lower()](**kwargs)


def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class GraphTripleConv(nn.Module):

    def __init__(self, input_dim, output_dim=None, hidden_dim=512,
                 pooling='avg', mlp_normalization='none'):
        super(GraphTripleConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling
        net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
        self.net1.apply(_init_weights)

        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
        self.net2.apply(_init_weights)

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

        if self.pooling == 'avg':
            # Figure out how many times each object has appeared, again using
            # some scatter_add trickery.
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

    def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg',
                 mlp_normalization='none'):
        super(GraphTripleConvNet, self).__init__()

        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'pooling': pooling,
            'mlp_normalization': mlp_normalization,
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
    sampled = F.grid_sample(img_in, grid)  # (O, D, H, W)

    out = _pool_samples(sampled, obj_to_img, pooling=pooling)

    return out


def masks_to_layout(vecs, boxes, masks, obj_to_img, H, W=None, pooling='sum'):
    O, D = vecs.size()
    M = masks.size(1)
    assert masks.size() == (O, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)  # (O,256,256,2)

    img_in = vecs.view(O, D, 1, 1) * masks.float().view(O, 1, M, M)  # (O,128,16,16)
    sampled = F.grid_sample(img_in, grid)  # (O,128,256,256)
    out = _pool_samples(sampled, obj_to_img, pooling=pooling)  # (batch_size(1),128,256,256)
    return out


def _boxes_to_grid(boxes, H, W):
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]  # (O,1,1)
    x1, y1 = boxes[:, 2], boxes[:, 3]  # (O,1,1)
    ww = x1 - x0
    hh = y1 - y0

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)  # (1,1,256)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)  # (1,256,1)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)  # *2-1

    return grid


def _pool_samples(samples, obj_to_img, pooling='sum'):
    dtype, device = samples.dtype, samples.device
    O, D, H, W = samples.size()
    N = obj_to_img.data.max().item() + 1

    # Use scatter_add to sum the sampled outputs for each image
    out = torch.zeros(N, D, H, W, dtype=dtype, device=device)  # (7,128,128,128)
    idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)  # (42,128,128,128)
    out = out.scatter_add(0, idx, samples)

    if pooling == 'avg':
        # Divide each output mask by the number of objects; use scatter_add again
        # to count the number of objects per image.
        ones = torch.ones(O, dtype=dtype, device=device)
        obj_counts = torch.zeros(N, dtype=dtype, device=device)
        obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
        print(obj_counts)
        obj_counts = obj_counts.clamp(min=1)
        out = out / obj_counts.view(N, 1, 1, 1)
    elif pooling != 'sum':
        raise ValueError('Invalid pooling "%s"' % pooling)

    return out


class RefinementModule(nn.Module):
    def __init__(self, layout_dim, input_dim, output_dim,
                 normalization='instance', activation='leakyrelu'):
        super(RefinementModule, self).__init__()

        layers = []
        layers.append(nn.Conv2d(layout_dim + input_dim, output_dim,
                                kernel_size=3, padding=1))
        layers.append(get_normalization_2d(output_dim, normalization))
        layers.append(get_activation(activation))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1))
        layers.append(get_normalization_2d(output_dim, normalization))
        layers.append(get_activation(activation))
        layers = [layer for layer in layers if layer is not None]
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
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
    def __init__(self, dims, normalization='instance', activation='leakyrelu'):
        super(RefinementNetwork, self).__init__()
        layout_dim = dims[0]
        self.refinement_modules = nn.ModuleList()
        for i in range(1, len(dims)):
            input_dim = 1 if i == 1 else dims[i - 1]
            output_dim = dims[i]
            mod = RefinementModule(layout_dim, input_dim, output_dim,
                                   normalization=normalization, activation=activation)
            self.refinement_modules.append(mod)

    def forward(self, layout):

        # H, W = self.output_size
        N, _, H, W = layout.size()
        self.layout = layout

        # Figure out size of input
        input_H, input_W = H, W  # 256→8
        for _ in range(len(self.refinement_modules)):
            input_H //= 2
            input_W //= 2

        assert input_H != 0
        assert input_W != 0

        feats = torch.zeros(N, 1, input_H, input_W).to(layout)  # (1,1,8,8)
        for mod in self.refinement_modules:
            feats = F.upsample(feats, scale_factor=2, mode='nearest')
            feats = mod(layout, feats)

        layout_features = feats  # (1,64,256,256)

        return layout_features


class Sg2ImModel(nn.Module):
    def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
                 gconv_dim=128, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5,
                 mask_size=None, mlp_normalization='none'):

        super(Sg2ImModel, self).__init__()

        self.vocab = vocab
        self.image_size = image_size

        num_objs = len(vocab['object_classes_itn'])
        num_relation = len(vocab['relationship_classes_itn'])

        self.obj_embeddings = nn.Embedding(num_objs, embedding_dim)
        self.pred_embeddings = nn.Embedding(num_relation, embedding_dim)

        if gconv_num_layers == 0:
            self.gconv = nn.Linear(embedding_dim, gconv_dim)
        elif gconv_num_layers > 0:
            gconv_kwargs = {
                'input_dim': embedding_dim,
                'output_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
                'input_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers - 1,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        box_net_dim = 4
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

        self.mask_net = None
        if mask_size is not None and mask_size > 0:
            self.mask_net = self._build_mask_net(num_objs, gconv_dim, mask_size)

        # rel_aux_layers = [2 * embedding_dim + 8, gconv_hidden_dim, num_relation]
        # self.rel_aux_net = build_mlp(rel_aux_layers, batch_norm=mlp_normalization)

        # refinement_kwargs = {
        #     'dims': (gconv_dim + layout_noise_dim,) + refinement_dims,
        #     'normalization': normalization,
        #     'activation': activation,
        # }
        # self.refinement_net = RefinementNetwork(**refinement_kwargs)

        # image_input (batch_size(1),3,64,64)
        image_layers = [nn.Conv2d(3, 64, 3, 1, 1),
                        nn.ReLU()]

        self.image_net = nn.Sequential(*image_layers)

    def _build_mask_net(self, num_objs, dim, mask_size):
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

    def forward(self, img, objs, triples, obj_to_img=None,
                boxes_gt=None, masks_gt=None):

        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]
        edges = torch.stack([s, o], dim=1)

        if obj_to_img is None:
            obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

        obj_vecs = self.obj_embeddings(objs)
        # obj_vecs_orig = obj_vecs
        pred_vecs = self.pred_embeddings(p)

        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)  # GraphTripleConvNet forward函数

        boxes_pred = self.box_net(obj_vecs)  # (O,4)

        masks_pred = None
        if self.mask_net is not None:
            mask_scores = self.mask_net(obj_vecs.view(O, -1, 1, 1))  # (O,128,1,1)→(O,1,16,16)
            masks_pred = mask_scores.squeeze(1).sigmoid()  # (O,1,16,16)→(O,16,16)

        # s_boxes, o_boxes = boxes_pred[s], boxes_pred[o]
        # s_vecs, o_vecs = obj_vecs_orig[s], obj_vecs_orig[o]
        # rel_aux_input = torch.cat([s_boxes, o_boxes, s_vecs, o_vecs], dim=1)
        # rel_scores = self.rel_aux_net(rel_aux_input)

        H, W = self.image_size
        layout_boxes = boxes_pred if boxes_gt is None else boxes_gt  # (1,O,4)

        if masks_pred is None:
            layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
        else:
            layout_masks = masks_pred if masks_gt is None else masks_gt
            layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks,
                                     obj_to_img, H, W)  # (batch_size(1),128,64,64)

        # if self.layout_noise_dim > 0:
        #     N, C, H, W = layout.size()
        #     noise_shape = (N, self.layout_noise_dim, H, W)
        #     layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
        #                                device=layout.device)
        #     layout = torch.cat([layout, layout_noise], dim=1)

        # layout_features = self.refinement_net(layout)  # (batch_size(1),64,64,64)

        image_features = self.image_net(img)  # (batch_size(1),64,64,64)

        features=torch.cat((layout,image_features),dim=1) #(1,128+64,64,64)

        # if i==0:
        #     features=image_features#(batch_size,64,64,64)
        #
        # else:
        #     features=image_features*0.5+layout_features*0.5


        #features = torch.cat([layout_features, image_features], dim=1) #(batch_size,128,64,64)

        return features, boxes_pred


# class ConcatConv2d(nn.Module):
#
#     def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, bias=True):
#         super(ConcatConv2d, self).__init__()
#         module = nn.Conv2d
#         self._layer = module(
#             dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, bias=bias)
#
#     def forward(self, t, x):
#         tt = torch.ones_like(x[:, :1, :, :]) * t
#         ttx = torch.cat([tt, x], 1)  # N C W H
#         return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim1,dim2):
        super(ODEfunc, self).__init__()
        self.conv11 = nn.Conv2d(dim1, dim1*2, 3, 1, 1)
        self.conv12 = nn.Conv2d(dim1*2, dim1, 3, 1, 1)

        self.conv21 = nn.Conv2d(dim2, dim2*2, 3, 1, 1)
        self.conv22 = nn.Conv2d(dim2*2, dim2, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        x1=x[:,0:128,:,:]#(1,128,64,64)
        x2=x[:,128:,:,:]#(1,64,64,64)
        #x1 layout(batch_size,128,64,64) x2 image_features(batch_size,64,64,64)
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


class FrameModel(nn.Module):
    def __init__(self, dims, normalization='instance', activation='leakyrelu'):#dims=128
        super(FrameModel, self).__init__()
        hidden_dims = dims // 2
        output_conv_layers = [
            nn.Conv2d(dims, hidden_dims, kernel_size=3, padding=1),
            get_activation(activation),
            nn.Conv2d(hidden_dims, hidden_dims, kernel_size=3, padding=1),
            get_activation(activation),
            nn.Conv2d(hidden_dims, 3, kernel_size=1, padding=0)  # output color image
        ]
        nn.init.kaiming_normal_(output_conv_layers[0].weight)
        nn.init.kaiming_normal_(output_conv_layers[2].weight)
        nn.init.kaiming_normal_(output_conv_layers[4].weight)
        self.output_conv = nn.Sequential(*output_conv_layers)

    def forward(self, features_new):
        out = self.output_conv(features_new)
        return out


class sgODE_model(nn.Module):

    def __init__(self, args, vocab, device):
        super(sgODE_model, self).__init__()
        self.args = args
        self.vocab = vocab
        self.device = device
        self.build_netG()

    def build_netG(self):
        sg_kwargs = {
            'vocab': self.vocab,
            'image_size': self.args.image_size,
            'embedding_dim': self.args.embedding_dim,
            'gconv_dim': self.args.gconv_dim,
            'gconv_hidden_dim': self.args.gconv_hidden_dim,
            'gconv_num_layers': self.args.gconv_num_layers,
            'mlp_normalization': self.args.mlp_normalization,
            'mask_size': self.args.mask_size
        }
        self.feature_model = Sg2ImModel(**sg_kwargs).to(self.device)

        self.odefunc = ODEfunc(dim1=128,dim2=64).to(self.device)

        frame_kwargs = {'dims':128,
                        'normalization': 'instance',
                        'activation': 'leakyrelu'

                        }
        self.frame_model = FrameModel(**frame_kwargs).to(self.device)

        refinement_dims = (1024, 512, 256, 128, 64)
        refinement_kwargs = {
            'dims': (128 + self.args.layout_noise_dim,) + refinement_dims,  #tuple(160,1024,512,256,128,64)
            'normalization': 'group',
            'activation': 'leakyrelu-0.2',
        }

        self.refinement_net = RefinementNetwork(**refinement_kwargs).to(self.device)

    def frame_generate(self, img, objs, triples, obj_to_img=None,
                       boxes_gt=None, masks_gt=None, time_stamps=None):

        features, boxes_pred = self.feature_model(img, objs, triples, obj_to_img, boxes_gt, masks_gt)
        # features(layout(N,128,64,64),image_features(N,64,64,64)) boxes_pred(O,4)

        # self.rel_scores=rel_scores
        self.boxes_gt = boxes_gt
        # self.relationship = triples[:, 1]

        self.time_stamps = time_stamps.type_as(features)
        features_new = odeint(self.odefunc, features, self.time_stamps, rtol=self.args.tol, atol=self.args.tol)
        # features_new(steps,batchsize,128,64,64)

        features_new = features_new.squeeze()# (generated_frame_num,1,192,64,64)→(num,192,64,64)

        layout=features_new[:,0:128,:,:]#(num,128,64,64)
        image_features=features_new[:,128:,:,:]#(num,64,64,64)

        # frame_pred_from_image = self.frame_model(image_features)#(num,3,64,64)

        self.layout_noise_dim = self.args.layout_noise_dim
        N, C, H, W = layout.size()
        noise_shape = (N, self.layout_noise_dim, H, W)#(num,32,64,64)
        layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
                                   device=layout.device)
        layout = torch.cat([layout, layout_noise], dim=1)#(num,160,64,64)
        layout_features = self.refinement_net(layout)  # (num,64,64,64)

        # features_combine=layout_features*0.5+image_features*0.5
        features_combine=torch.cat((layout_features,image_features),dim=1)#(num,128,64,64)

        frame_pred = self.frame_model(features_combine)#(num,3,64,64)
        # self.frame_pred=frame_pred

        return frame_pred, boxes_pred


def compute_losses(args, boxes_pred, boxes_gt,frame_pred,frame_real):
    loss_bbox = F.mse_loss(boxes_pred, boxes_gt) * args.bbox_loss_weight
    # loss_relationship = F.cross_entropy(self.rel_scores,self.relationship)*args.relationship_loss_weight
    loss_img = F.l1_loss(frame_pred, frame_real)*args.img_loss_weight
    # loss_iimg =F.l1_loss(frame_pred_from_image, frame_real)*args.iimg_loss_weight
    total_loss=loss_bbox+loss_img

    return total_loss

class Discriminator(nn.Module):#DCGAN
    def __init__(self, input_channel=3, hidden=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input batch_size x 3 x 64 x 64
            nn.Conv2d(input_channel, hidden, 4, 2, 1, bias=False),# 32 x 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, hidden * 2, 4, 2, 1, bias=False),# 16 x 16
            nn.GroupNorm(32,hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden * 2, hidden * 4, 4, 2, 1, bias=False),# 8 x 8
            nn.GroupNorm(32,hidden * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden * 4, hidden * 8, 4, 2, 1, bias=False),# 4 x 4
            nn.GroupNorm(32,hidden * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            #state size 1 x 1 x 1
        )
        self.disc.apply(weights_init)


    def forward(self, input):
        return self.disc(input)

# class Discriminator(nn.Module):  # DCGAN
#     def __init__(self, input_channel=3, hidden=16):
#         super(Discriminator, self).__init__()
#         self.disc = nn.Sequential(
#             # input: N x 3 x 64 x 64
#             nn.Conv2d(
#                 input_channel, hidden, kernel_size=4, stride=2, padding=1
#             ),  # N x 16 x 32 x 32
#             nn.LeakyReLU(0.2),
#             # _block(in_channels, out_channels, kernel_size, stride, padding)
#             self._block(hidden, hidden * 2, 4, 2, 1),  # 16 x 16
#             self._block(hidden * 2, hidden * 4, 4, 2, 1),  # 8 x 8
#             self._block(hidden * 4, hidden * 8, 4, 2, 1),  # 4 x 4
#             nn.Conv2d(hidden * 8, 1, kernel_size=4, stride=2, padding=0),  # N x 1 x 1 x 1
#             nn.Sigmoid(),
#         )

        # def forward(self, x):
        #     return self.disc(x)

    # def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    #     return nn.Sequential(
    #         nn.Conv2d(
    #             in_channels,
    #             out_channels,
    #             kernel_size,
    #             stride,
    #             padding,
    #             bias=False,
    #         ),
    #         nn.LeakyReLU(0.2),
    #     )
    #
    # def forward(self, x):
    #     return self.disc(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # mean=0,stdev=0.02
    elif classname.find('GroupNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)












