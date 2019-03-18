import torch 
import torchvision
import torch.nn as nn
from utils import *
import numpy as np
import os
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import tensorflow as tf
# import cv2
import numpy as np
from PIL import Image
import onnx
from onnx_tf.backend import prepare
import pdb
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# Resnet Backbone
# resnet34 = torchvision.models.resnet34(pretrained=False)
# # The model with the ResnNet backbone and without 
# # the top classification layers
# backbone = nn.Sequential(*list(resnet34.children())[:-2])



grids = [4, 2, 1] 
zooms = [0.7, 1., 1.3]
ratios = [[1., 1.], [1., 0.5], [0.5, 1.]]
drop = 0.3
bais = -4
classes = ['Green',
			'Red',
			'Yellow',
			'GreenLeft',
			'RedLeft',
			'RedRight',
			'RedStraight',
			'RedStraightLeft',
			'GreenRight',
			'GreenStraightRight',
			'GreenStraightLeft',
			'GreenStraight',
			'off']

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class SingleShotDetector(object):
    
    def __init__(self, backbone, model_path, classes, grids=[4, 2, 1], zooms=[0.7, 1., 1.3], ratios=[[1., 1.], [1., 0.5], [0.5, 1.]]
    	, drop=0.3, bias=-4.):
        
        super().__init__()

        self._device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

            
        self._create_anchors(grids, zooms, ratios)
        
        # ssd_head = SSDHead(grids, self._anchors_per_cell, len(classes) + 1, drop=drop, bias=bias)

        # self.model = nn.Sequential(backbone, ssd_head)
        # self.learn.model = self.learn.model.to(self._device)

        # Get the model ready for inception 
        # for param in self.model.parameters():
        #     param.requires_grad = False
       	
        # state_dict = torch.load(model_path)
        # self.model.load_state_dict(state_dict['model'], strict=False)

            
        # self.learn.loss_func = self._ssd_loss

    @classmethod
    def from_emd(cls, data, emd_path):
        emd = json.load(open(emd_path))
        class_mapping = {i['Value'] : i['Name'] for i in emd['Classes']}
        if data is None:
            empty_data = _EmptyData(path='str', loss_func=None, c=len(class_mapping) + 1)
            return cls(empty_data, emd['Grids'], emd['Zooms'], emd['Ratios'], pretrained_path=emd['ModelFile'])
        else:
            return cls(data, emd['Grids'], emd['Zooms'], emd['Ratios'], pretrained_path=emd['ModelFile'])

        
    def _create_anchors(self, anc_grids, anc_zooms, anc_ratios):
        
        self.grids = anc_grids
        self.zooms = anc_zooms
        self.ratios =  anc_ratios

        anchor_scales = [(anz*i, anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
        
        self._anchors_per_cell = len(anchor_scales)
        
        anc_offsets = [1/(o*2) for o in anc_grids]

        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), self._anchors_per_cell, axis=0)

        anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                       for ag in anc_grids])
        
        self._grid_sizes = torch.Tensor(np.concatenate([np.array([ 1/ag  for i in range(ag*ag) for o,p in anchor_scales])
                       for ag in anc_grids])).unsqueeze(1).to(self._device)
        
        self._anchors = torch.Tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float().to(self._device)
        
        self._anchor_cnr = self._hw2corners(self._anchors[:,:2], self._anchors[:,2:])
        
    def _hw2corners(self, ctr, hw): 
        return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)

    def _get_y(self, bbox, clas):
        bbox = bbox.view(-1,4) #/sz
        bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
        return bbox[bb_keep],clas[bb_keep]

    def _actn_to_bb(self, actn, anchors, grid_sizes):
        actn_bbs = torch.tanh(actn)
        actn_centers = (actn_bbs[:,:2]/2 * grid_sizes) + anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * anchors[:,2:]
        return self._hw2corners(actn_centers, actn_hw)

    def _map_to_ground_truth(self, overlaps, print_it=False):
        prior_overlap, prior_idx = overlaps.max(1)
        if print_it: print(prior_overlap)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i,o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap, gt_idx
        
        
    def _ssd_1_loss(self, b_c, b_bb, bbox, clas, print_it=False):
        bbox,clas = self._get_y(bbox,clas)
        bbox = self._normalize_bbox(bbox)

        a_ic = self._actn_to_bb(b_bb, self._anchors, self._grid_sizes)
        overlaps = self._jaccard(bbox.data, self._anchor_cnr.data)
        try:
            gt_overlap,gt_idx = self._map_to_ground_truth(overlaps,print_it)
        except Exception as e:
            return 0.,0.
        gt_clas = clas[gt_idx]
        pos = gt_overlap > 0.4
        pos_idx = torch.nonzero(pos)[:,0]
        gt_clas[1-pos] = 0 #data.c - 1 # CHANGE
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
        clas_loss  = self._loss_f(b_c, gt_clas)
        return loc_loss, clas_loss
    
    # def _ssd_loss(self, pred, targ1, targ2, print_it=False):
    #     lcs,lls = 0.,0.
    #     for b_c,b_bb,bbox,clas in zip(*pred, targ1, targ2):
    #         loc_loss,clas_loss = self._ssd_1_loss(b_c,b_bb,bbox.cuda(),clas.cuda(),print_it)
    #         lls += loc_loss
    #         lcs += clas_loss
    #     if print_it: print(f'loc: {lls}, clas: {lcs}') #CHANGE
    #     return lls+lcs
    
    def _intersect(self,box_a, box_b):
        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def _box_sz(self, b): 
        return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

    def _jaccard(self, box_a, box_b):
        inter = self._intersect(box_a, box_b)
        union = self._box_sz(box_a).unsqueeze(1) + self._box_sz(box_b).unsqueeze(0) - inter
        return inter / union
    
    def _normalize_bbox(self, bbox): 
        return (bbox+1.)/2.





# print(backbone)
# ssd = SingleShotDetector(backbone, './models/ssd-tl-FL-715.pth', classes)
ssd = SingleShotDetector('', './models/ssd-tl-FL-715.pth', '')

# dummy_input = torch.randn(2, 3, 224, 224)
# # Preprocessing
# output = ssd.model(dummy_input)
# torch.save(ssd.model, './models/pytorch_model_715')


def nms(boxes, scores, overlap=0.5, top_k=100):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0: return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1: break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        print(count)
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count



def analyze_pred(pred, thresh=0.5, nms_overlap=0.1, ssd=None):
    print('Heeeey, Im heeerere')
#         pdb.set_trace()
    # def analyze_pred(pred, anchors, grid_sizes, thresh=0.5, nms_overlap=0.1, ssd=None):
    b_clas, b_bb = pred
    # pdb.set_trace()
    a_ic = ssd._actn_to_bb(b_bb, ssd._anchors.cpu(), ssd._grid_sizes.cpu())

    conf_scores, clas_ids = b_clas[:, 1:].max(1)
    conf_scores = b_clas.t().sigmoid()
    # pdb.set_trace()

    out1, bbox_list, class_list = [], [], []

    for cl in range(1, len(conf_scores)):
        c_mask = conf_scores[cl] > thresh
        if c_mask.sum() == 0: 
            print("Skipped{}".format(c_mask.sum()))
            continue
        scores = conf_scores[cl][c_mask]
        l_mask = c_mask.unsqueeze(1)
        l_mask = l_mask.expand_as(a_ic)
        boxes = a_ic[l_mask].view(-1, 4) # boxes are now in range[ 0, 1]
        boxes = (boxes-0.5) * 2.0        # putting boxes in range[-1, 1]
        ids, count = nms(boxes.data, scores, nms_overlap, 50) # FIX- NMS overlap hardcoded
        ids = ids[:count]
        out1.append(scores[ids])
        bbox_list.append(boxes.data[ids])
        class_list.append(torch.tensor([cl]*count))
    # pdb.set_trace()
    if len(bbox_list) == 0:
        return None #torch.Tensor(size=(0,4)), torch.Tensor()

    return torch.cat(bbox_list, dim=0), torch.cat(class_list, dim=0) # torch.cat(out1, dim=0), 


model = torch.load('./models/pytorch_model_715')#.cuda()
# print(model)
normalize = transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1])
preprocessor = torchvision.transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])

img = Image.open('./TESTJPEGS/20340.jpg')
# # print(img.size)
# # img.show()
# # print( np.asarray(img).shape)
# # Resized Normalized Tensor
t_image = preprocessor(img).float()
# # print('normalization')
# # print(t_image.numpy()[0][0])
# # t_image = Variable(t_image, requires_grad=False)
# # print(t_image.size())
# # print(t_image.unsqueeze(0).size())
# # Prepare the model for inception
model.eval()
# # Predict t_image
output = model(t_image.unsqueeze(0))
# print("pytorch ouput \n{}".format(output))
output[0] = output[0].squeeze(0)
output[1] = output[1].squeeze(0)
# print(output)
# SAVE FILE
# outfile = TemporaryFile()

#SAVE tensor
# np.savez('tensor', preds_0=output[0], preds_1=output[1])
# outfile.seek(0)
a = analyze_pred(output, thresh=0.3, ssd=ssd)
print(a)
# npzfile = np.load('tensor.npz')
# print(npzfile.files)
# print(npzfile['preds_0'])
# outfile.seek()
# print(np.expand_dims(t_image.numpy(), axis=0).shape)
# onnx model
# onnx_model = onnx.load("model_715.onnx")
# tf_rep = prepare(onnx_model)
# output_onnx_tf = tf_rep.run(np.expand_dims(t_image.numpy(), axis=0))
# print(output_onnx_tf)





# print(model)


# From Pytorch to Keras
# we should specify shape of the input tensor
# input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
# input_var = Variable(torch.FloatTensor(input_np))
# k_model = pytorch_to_keras(model,
#                          input_var,
#                          [(3, 224, 224,)],
#                           verbose=True)  
# k_model.summary()


# From Pytorch to onnx 
# torch.onnx.export(model
#                  , torch.randn(1, 3, 224, 224).float()
#                  ,"model_715.onnx"
#                  , verbose=True)

# t_224 = torchvision.transforms.ToPILImage(t)



# output = model(dummy_input.unsqueeze(0))


# def _normalize_batch(b:Tensor,
#                      mean:FloatTensor,
#                      std:FloatTensor,
#                      do_x:bool=True,
#                     do_y:bool=False)->Tensor:
#     "`b` = `x`,`y` - normalize `x` array of imgs and `do_y` optionally `y`."
#     x = b
#     mean,std = mean.to(x.device),std.to(x.device)
#     return (x-mean[...,None,None]) / std[...,None,None]


# print(len(output[0]))
# print(output[0][0])
# head = SSDHead(grids, self._a)
