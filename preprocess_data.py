import numpy as np
import tensorflow as tf
import typing

class BboxAnalyzer(object):
	def __init__(self,grids=[4, 2, 1], zooms=[0.7, 1., 1.3],ratios=[[1., 1.], [1., 0.5], [0.5, 1.]], bias=-4.):
		super().__init__()
		self._create_anchors(grids, zooms, ratios)

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

	    anc_sizes  =  np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
	                   for ag in anc_grids])
	    
	    # self._grid_sizes = torch.Tensor(np.concatenate([np.array([ 1/ag  for i in range(ag*ag) for o,p in anchor_scales])
	    #                for ag in anc_grids])).unsqueeze(1)#.to(self._device)
	    self._grid_sizes = tf.expand_dims(tf.constant(np.concatenate([np.array([ 1/ag  for i in range(ag*ag) for o,p in anchor_scales])
	                   for ag in anc_grids])), 1)
	    #self._anchors = torch.Tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float()#.to(self._device)
	    self._anchors = tf.constant(np.concatenate([anc_ctrs, anc_sizes], axis=1), dtype=tf.float32)
	    self._anchor_cnr = self._hw2corners(self._anchors[:,:2], self._anchors[:,2:])
	
	def _hw2corners(self, ctr, hw):
		return tf.concat([ctr-hw/2, ctr+hw/2], 1)

	def analyze_pred(self, pred, thresh=0.5, nms_overlap=0.1, ssd=None):
		print('Analyzing predictions ................................./')
		b_clas, b_bb = pred
		a_ic = self._actn_to_bb(b_bb, self._anchors, self._grid_sizes)

		# TODO : I don't get it 
		conf_scores, clas_ids = b_clas[:, 1:].max(1)
		conf_scores = b_clas.t().sigmoid()

		out1, bbox_list, class_list = [], [], []

		for cl in range(1, len(conf_scores)):
			c_mask = conf_scores[cl] > thresh
			if c_mask.sum() == 0: 
				continue
			scores = conf_scores[cl][c_mask]
			l_mask = tf.expand_dims(c_mask, 1)#.unsqueeze(1)
			l_mask = tf.contrib.framework.broadcast_to(l_mask, a_ic.shape)#l_mask.expand_as(a_ic)
			boxes = a_ic[l_mask].view(-1, 4) # boxes are now in range[ 0, 1]
			boxes = (boxes-0.5) * 2.0        # putting boxes in range[-1, 1]
			ids, count = nms(boxes.data, scores, nms_overlap, 50) # FIX- NMS overlap hardcoded
			ids = ids[:count]
			out1.append(scores[ids])
			bbox_list.append(boxes.data[ids])
			class_list.append(torch.tensor([cl]*count))

		if len(bbox_list) == 0:
			return None #torch.Tensor(size=(0,4)), torch.Tensor()

		return torch.cat(bbox_list, dim=0), torch.cat(class_list, dim=0) # torch.cat(out1, dim=0), 

    
	def _actn_to_bb(self, actn, anchors, grid_sizes):
		actn_bbs = tf.math.tanh(actn)
		actn_centers = (actn_bbs[:,:2]/2 * grid_sizes) + anchors[:,:2]
		actn_hw = (actn_bbs[:,2:]/2+1) * anchors[:,2:]
		return self._hw2corners(actn_centers, actn_hw)