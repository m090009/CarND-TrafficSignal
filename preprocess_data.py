import numpy as np
import tensorflow as tf
import typing
import pdb

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
	    self._grid_sizes = tf.dtypes.cast(tf.expand_dims(tf.constant(np.concatenate([np.array([ 1/ag  for i in range(ag*ag) for o,p in anchor_scales])
	                   for ag in anc_grids])), 1), tf.float32)
	    #self._anchors = torch.Tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float()#.to(self._device)
	    self._anchors = tf.constant(np.concatenate([anc_ctrs, anc_sizes], axis=1), dtype=tf.float32)
	    self._anchor_cnr = self._hw2corners(self._anchors[:,:2], self._anchors[:,2:])
	
	def _hw2corners(self, ctr, hw):
		return tf.concat([ctr-hw/2, ctr+hw/2], 1)

	def analyze_pred(self, pred, sess,thresh=0.5, nms_overlap=0.1):
		print('Analyzing predictions ................................./')
		b_clas, b_bb = pred
		# pdb.set_trace()
		a_ic = self._actn_to_bb(b_bb, self._anchors, self._grid_sizes)
		# pdb.set_trace()
		# TODO : I don't get it 
		conf_scores = b_clas[:, 1:].max(1)
		conf_scores = tf.transpose(tf.math.sigmoid(tf.constant(b_clas)))


		out1, bbox_list, class_list = [], [], []

		for cl in range(1, conf_scores.get_shape().as_list()[0]):
			c_mask = conf_scores[cl] > thresh
			if sess.run(tf.reduce_sum(tf.dtypes.cast(c_mask, tf.int32)) ) == 0:
				print("Skipped")
				continue
			num = tf.reduce_sum(tf.dtypes.cast(c_mask, tf.int32))
			scores = tf.boolean_mask(conf_scores[cl], c_mask)
			l_mask = tf.expand_dims(c_mask, 1)#.unsqueeze(1)
			l_mask = tf.broadcast_to(l_mask, a_ic.shape)#l_mask.expand_as(a_ic)
			boxes = tf.reshape(tf.boolean_mask(a_ic, l_mask), [-1, 4])#tf.reshape(a_ic[l_mask], [-1, 4]) #a_ic[l_mask].view(-1, 4) # boxes are now in range[ 0, 1]
			boxes = (boxes-0.5) * 2.0        # putting boxes in range[-1, 1]

			ids, count = self.nms(boxes, num, scores, nms_overlap, 50, sess=sess) # FIX- NMS overlap hardcoded
			ids = ids[:count]
			out1.append(tf.gather(scores, ids))
			bbox_list.append(tf.gather(boxes, ids))
			class_list.append(tf.constant([cl]*count))
		# pdb.set_trace()
		if len(bbox_list) == 0:
			return None #torch.Tensor(size=(0,4)), torch.Tensor()

		# return torch.cat(bbox_list, dim=0), torch.cat(class_list, dim=0) # torch.cat(out1, dim=0), 
		return tf.concat(bbox_list, 0), tf.concat(class_list, 0)
    
	def _actn_to_bb(self, actn, anchors, grid_sizes):
		actn_bbs = tf.math.tanh(actn)
		actn_centers = (actn_bbs[:,:2]/2 * grid_sizes) + anchors[:,:2]
		actn_hw = (actn_bbs[:,2:]/2+1) * anchors[:,2:]
		return self._hw2corners(actn_centers, actn_hw)


	def nms(self, boxes, num, scores, overlap=0.5, top_k=100, sess=None):
		# keep = scores.new(scores.size(0)).zero_().long()
		keep = tf.Variable(tf.zeros(tf.size(scores), tf.int32))
		# TODO relook into it
		if tf.size(boxes)  == 0: return keep
		x1 = boxes[:, 0]
		y1 = boxes[:, 1]
		x2 = boxes[:, 2]
		y2 = boxes[:, 3]
		area = tf.multiply(x2 - x1, y2 - y1)
		
		idx = tf.argsort(scores, axis=0)#scores.sort(0)  # sort in ascending order
		idx = idx[-top_k:]  # indices of the top-k largest vals
		xx1 = tf.placeholder(boxes.dtype, shape=[None])
		yy1 = tf.placeholder(boxes.dtype, shape=[None])
		xx2 = tf.placeholder(boxes.dtype, shape=[None])
		yy2 = tf.placeholder(boxes.dtype, shape=[None])
		w = tf.placeholder(boxes.dtype, shape=[None])
		h = tf.placeholder(boxes.dtype, shape=[None])

		t_overlap = tf.constant(overlap)
		count = 0
		while (sess.run(tf.size(idx) > 0)):
			print(count)
			i = idx[-1]  # index of current largest val
			# m_mask = np.zeros(keep.get_shape().as_list()[0], dtype=np.int64)
			# m_mask[count] = i
			keep = tf.scatter_update(keep, tf.Variable(count), i)
			# keep = tf.add(keep, tf.constant(m_mask))
			#keep[count] = i
			count += 1
			if idx.get_shape().as_list()[0] == 1:  break
			idx = idx[:-1]  # remove kept element from view
			# load bboxes of next highest vals
			xx1 = tf.gather(x1, idx)
			yy1 = tf.gather(y1, idx)
			xx2 = tf.gather(x2, idx)
			yy2 = tf.gather(y2, idx)
			# store element-wise max with next highest score
			# xx1 = torch.clamp(xx1, min=x1[i])
			xx1 = tf.clip_by_value(xx1, x1[i], tf.reduce_max(xx1))
			# yy1 = torch.clamp(yy1, min=y1[i])
			yy1 = tf.clip_by_value(yy1, y1[i], tf.reduce_max(yy1))
			# xx2 = torch.clamp(xx2, max=x2[i])
			xx2 = tf.clip_by_value(xx2, tf.reduce_min(xx2), x2[i])
			# yy2 = torch.clamp(yy2, max=y2[i])
			yy2 = tf.clip_by_value(yy2, tf.reduce_min(yy2), y2[i])

			# w.resize_as_(xx2)
			w = w[:xx2.get_shape().as_list()[0]]
			# h.resize_as_(yy2)
			h = h[:yy2.get_shape().as_list()[0]]
			w = xx2 - xx1
			h = yy2 - yy1
			# check sizes of xx1 and xx2.. after each iteration
			w = tf.clip_by_value(w, 0.0, tf.reduce_max(w))
			h = tf.clip_by_value(h, 0.0, tf.reduce_max(h))
			inter = w*h
			# IoU = i / (area(a) + area(b) - i)
			rem_areas = tf.gather(area, idx)  # load remaining areas)
			union = (rem_areas - inter) + area[i]
			IoU = inter / union  # store result in iou
			# keep only elements with an IoU <= overlap
			# idx = idx[IoU.le(overlap)]
		
			idx = tf.boolean_mask(idx, tf.math.less_equal(IoU, t_overlap))
			# idx = tf.Variable(sess.run(tf.boolean_mask(idx, tf.math.less_equal(IoU, t_overlap))))



		# cond = tf.cond(tf.size(idx) > 0 , lambda: True, lambda:False)
		# loop_vars = [idx, keep, count, x1, x2, y1, y2, xx1, xx2, yy1, yy2, w, h, area, overlap]
		# def loop_cond(idx, keep, count, x1, x2, y1, y2, xx1, xx2, yy1, yy2, w, h, area, overlap):
		# 	return tf.size(idx) > 0
		# def loop_body(idx, keep, count, x1, x2, y1, y2, xx1, xx2, yy1, yy2, w, h, area, overlap):
		# 	i = idx[-1]  # index of current largest val
		# 	pdb.set_trace()
		# 	# m_mask = np.zeros(keep.get_shape().as_list()[0], dtype=np.int64)
		# 	# m_mask[count] = i
		# 	keep = tf.scatter_update(keep, count, i)
		# 	# keep = tf.add(keep, tf.constant(m_mask))
		# 	#keep[count] = i
		# 	count += 1
		# 	if idx.get_shape().as_list()[0] == 1: return keep, count
		# 	idx = idx[:-1]  # remove kept element from view
		# 	# load bboxes of next highest vals
		# 	xx1 = tf.gather(x1, idx)
		# 	yy1 = tf.gather(y1, idx)
		# 	xx2 = tf.gather(x2, idx)
		# 	yy2 = tf.gather(y2, idx)
		# 	# store element-wise max with next highest score
		# 	# xx1 = torch.clamp(xx1, min=x1[i])
		# 	xx1 = tf.clip_by_value(xx1, x1[i], tf.reduce_max(xx1))
		# 	# yy1 = torch.clamp(yy1, min=y1[i])
		# 	yy1 = tf.clip_by_value(yy1, y1[i], tf.reduce_max(yy1))
		# 	# xx2 = torch.clamp(xx2, max=x2[i])
		# 	xx2 = tf.clip_by_value(xx2, tf.reduce_min(xx2), x2[i])
		# 	# yy2 = torch.clamp(yy2, max=y2[i])
		# 	yy2 = tf.clip_by_value(yy2, tf.reduce_min(yy2), y2[i])

		# 	# w.resize_as_(xx2)
		# 	w = tf.reshape(w, xx2.shape)
		# 	# h.resize_as_(yy2)
		# 	h = tf.reshape(w, yy2.shape)
		# 	w = xx2 - xx1
		# 	h = yy2 - yy1
		# 	# check sizes of xx1 and xx2.. after each iteration
		# 	w = tf.clip_by_value(w, 0.0, tf.reduce_max(w))
		# 	h = tf.clip_by_value(h, 0.0, tf.reduce_max(h))
		# 	inter = w*h
		# 	# IoU = i / (area(a) + area(b) - i)
		# 	rem_areas = tf.gather(area, idx)  # load remaining areas)
		# 	union = (rem_areas - inter) + area[i]
		# 	IoU = inter / union  # store result in iou
		# 	# keep only elements with an IoU <= overlap
		# 	idx = idx[IoU.le(overlap)]
		# 	return idx, keep, count, xx1, xx2, yy1, yy2, w, h
		# pdb.set_trace()
		# idx, keep, count, xx1, xx2, yy1, yy2, w, h = tf.while_loop(loop_cond, loop_body, loop_vars)

		# pdb.set_trace()

		return keep, count

npzfile = np.load('tensor.npz')
preds = [npzfile['preds_0'], npzfile['preds_1']]
# print(preds)
preds_0 = tf.constant(preds[0], dtype=tf.float64)
preds_2 = tf.constant(preds[1], dtype=tf.float64)
analyzer = BboxAnalyzer()
sess = tf.InteractiveSession()  
out = analyzer.analyze_pred(preds, sess,thresh=0.3)
# print(out)
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(out))
# sess.run(out)