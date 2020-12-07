#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:46:21 2020

@author: worklab
"""


import os,sys,inspect
import numpy as np
import random 
import time
random.seed = 0
import cv2
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision.ops import roi_align
import matplotlib.pyplot  as plt
from scipy.optimize import linear_sum_assignment
import _pickle as pickle
import csv


# filter and frame loader
from util_track.mp_loader import FrameLoader
from util_track.kf import Torch_KF
from util_track.mp_writer import OutputWriter



class Localization_Tracker():
    
    def __init__(self,
                 track_dir,
                 detector,
                 localizer,
                 kf_params,
                 class_dict,
                 det_step = 1,
                 init_frames = 1,
                 fsld_max = 1,
                 matching_cutoff = 100,
                 iou_cutoff = 0.5,
                 det_conf_cutoff = 0.5,
                 ber = 2.0,
                 PLOT = True,
                 OUT = None,
                 wer = 1.25,
                 skip_step = 1,
                 checksum_path = None,
                 geom_path = None,
                 output_dir):
        """
         Parameters
        ----------
        track_dir : str
            path to directory containing ordered track images
        detector : object detector with detect function implemented that takes a frame and returns detected object
        localizer : CNN object localizer
        kf_params : dictionaryh
            Contains the parameters to initialize kalman filters for tracking objects
        det_step : int optional
            Number of frames after which to perform full detection. The default is 1.
        init_frames : int, optional
            Number of full detection frames before beginning localization. The default is 3.
        fsld_max : int, optional
            Maximum dense detection frames since last detected before an object is removed. 
            The default is 1.
        matching_cutoff : int, optional
            Maximum distance between first and second frame locations before match is not considered.
            The default is 100.
        iou_cutoff : float in range [0,1], optional
            Max iou between two tracked objects before one is removed. The default is 0.5.       
        ber : float, optional
            How much bounding boxes are expanded before being fed to localizer. The default is 1.
        PLOT : bool, optional
            If True, resulting frames are output. The default is True. 
        """
        
        self.input_file_name = track_dir # either folder of frames or video
        self.output_dir = output_dir
        
        #store parameters
        self.d = det_step
        self.s = skip_step
        self.init_frames = init_frames
        self.fsld_max = fsld_max
        self.matching_cutoff = matching_cutoff
        self.iou_cutoff = iou_cutoff
        self.det_conf_cutoff = det_conf_cutoff
        self.ber = ber
        self.PLOT = PLOT
        self.wer = wer
        self.state_size = kf_params["Q"].shape[0]
        
        # CUDA
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.cuda.empty_cache() 
       
        # store detector and localizer
        try:
            self.localizer = localizer.to(self.device)
            localizer.eval()
        except:
            self.localizer = None
        
        self.detector = detector.to(self.device)
        detector.eval()
       
        # store filter params
        self.filter = Torch_KF(torch.device("cpu"),INIT = kf_params)
       
        self.loader = FrameLoader(track_dir,self.device,buffer_size = 9,timestamp_checksum_path = checksum_path,timestamp_geom_path = geom_path)
        #self.track_id = int(track_dir.split("MVI_")[-1])
        
        # create output image writer
        if OUT is not None:
            self.writer = OutputWriter(OUT)
        else:
            self.writer = None
        
        time.sleep(5)
        self.n_frames = 10000 # expected number of frames, can be changed
        self.frames_processed = 0
    
        self.next_obj_id = 0             # next id for a new object (incremented during tracking)
        self.fsld = {}                   # fsld[id] stores frames since last detected for object id
    
        self.all_tracks = {}             # stores states for each object
        self.all_classes = {}            # stores class evidence for each object
        self.all_timestamps = []         # store timestamp at each frame
    
        self.class_dict = class_dict
    
        # for keeping track of what's using time
        self.time_metrics = {            
            "load":0,
            "predict":0,
            "pre_localize and align":0,
            "localize":0,
            "post_localize":0,
            "detect":0,
            "parse":0,
            "match":0,
            "update":0,
            "add and remove":0,
            "store":0,
            "plot":0
            }
    
    def manage_tracks(self,detections,matchings,pre_ids):
        """
        Updates each detection matched to an existing tracklet, adds new tracklets 
        for unmatched detections, and increments counters / removes tracklets not matched
        to any detection
        """
        start = time.time()

        # update tracked and matched objects
        update_array = np.zeros([len(matchings),4])
        update_ids = []
        for i in range(len(matchings)):
            a = matchings[i,0] # index of pre_loc
            b = matchings[i,1] # index of detections
           
            update_array[i,:] = detections[b,:4]
            update_ids.append(pre_ids[a])
            self.fsld[pre_ids[a]] = 0 # fsld = 0 since this id was detected this frame
        
        if len(update_array) > 0:    
            self.filter.update2(update_array,update_ids)
          
            self.time_metrics['update'] += time.time() - start
              
        
        # for each detection not in matchings, add a new object
        start = time.time()
        
        new_array = np.zeros([len(detections) - len(matchings),4])
        new_ids = []
        cur_row = 0
        for i in range(len(detections)):
            if len(matchings) == 0 or i not in matchings[:,1]:
                
                new_ids.append(self.next_obj_id)
                new_array[cur_row,:] = detections[i,:4]

                self.fsld[self.next_obj_id] = 0
                self.all_tracks[self.next_obj_id] = np.zeros([self.n_frames,self.state_size])
                self.all_classes[self.next_obj_id] = np.zeros(13)
                
                self.next_obj_id += 1
                cur_row += 1
       
        if len(new_array) > 0:        
            self.filter.add(new_array,new_ids)
        
        
        # 7a. For each untracked object, increment fsld        
        for i in range(len(pre_ids)):
            try:
                if i not in matchings[:,0]:
                    self.fsld[pre_ids[i]] += 1
            except:
                self.fsld[pre_ids[i]] += 1
        
        # 8a. remove lost objects
        removals = []
        for id in pre_ids:
            if self.fsld[id] >= self.fsld_max:
                removals.append(id)
                self.fsld.pop(id,None) # remove key from fsld
        if len(removals) > 0:
            self.filter.remove(removals)    
    
        self.time_metrics['add and remove'] += time.time() - start
    
    
    def crop_tracklets(self,pre_locations,frame):
        """
        Crops relevant areas from frame based on a priori (pre_locations) object locations
        """
        start = time.time()
        box_ids = []
        box_list = []
        
        # convert to array
        for id in pre_locations:
            box_ids.append(id)
            box_list.append(pre_locations[id][:4])
        boxes = np.array(box_list)
   
        temp = np.zeros(boxes.shape)
        temp[:,0] = (boxes[:,0] + boxes[:,2])/2.0
        temp[:,1] = (boxes[:,1] + boxes[:,3])/2.0
        temp[:,2] =  boxes[:,2] - boxes[:,0]
        temp[:,3] = (boxes[:,3] - boxes[:,1])/temp[:,2]
        boxes = temp
    
        # convert xysr boxes into xmin xmax ymin ymax
        # first row of zeros is batch index (batch is size 0) for ROI align
        new_boxes = np.zeros([len(boxes),5]) 

        # use either s or s x r for both dimensions, whichever is smaller,so crop is square
        box_scales = np.min(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1) #/2.0
            
        #expand box slightly
        box_scales = box_scales * self.ber# box expansion ratio
        
        new_boxes[:,1] = boxes[:,0] - box_scales/2
        new_boxes[:,3] = boxes[:,0] + box_scales/2 
        new_boxes[:,2] = boxes[:,1] - box_scales/2 
        new_boxes[:,4] = boxes[:,1] + box_scales/2 
        torch_boxes = torch.from_numpy(new_boxes).float().to(self.device)
    
        # crop using roi align 
        crops = roi_align(frame.unsqueeze(0),torch_boxes,(224,224))
        self.time_metrics['pre_localize and align'] += time.time() - start
        
        return crops,new_boxes,box_ids,box_scales
    
    
    def test_outputs(self,bboxes,crops):
        """
        Description
        -----------
        Generates a plot of the bounding box predictions output by the localizer so
        performance of this component can be visualized
        
        Parameters
        ----------
        bboxes - tensor [n,4] 
            bounding boxes output for each crop by localizer network
        crops - tensor [n,3,width,height] (here width and height are both 224)
        """
        
        # define figure subplot grid
        batch_size = len(crops)
        row_size = min(batch_size,8)
        fig, axs = plt.subplots((batch_size+row_size-1)//row_size, row_size, constrained_layout=True)
        
        for i in range(0,len(crops)):    
            # get image
            im   = crops[i].data.cpu().numpy().transpose((1,2,0))
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            im   = std * im + mean
            im   = np.clip(im, 0, 1)
            
            # get predictions
            bbox = bboxes[i].data.cpu().numpy()
            
            wer = self.wer
            imsize = 224
            
            # transform bbox coords back into im pixel coords
            bbox = (bbox* imsize*wer - imsize*(wer-1)/2).astype(int)
            # plot pred bbox
            im = cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0.1,0.6,0.9),2)
            im = im.get()
    
            if batch_size <= 8:
                axs[i].imshow(im)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
            else:
                axs[i//row_size,i%row_size].imshow(im)
                axs[i//row_size,i%row_size].set_xticks([])
                axs[i//row_size,i%row_size].set_yticks([])
            plt.pause(.001)    
          
            
    def local_to_global(self,reg_out,box_scales,new_boxes):
        """
        reg_out - tensor of shape [n_crops, n_anchors, 4]
        box_scales - tensor of shape [n_crops]
        new_boxes - tensor of shape n_crops,4
        """
        detections = reg_out
        # detections = (reg_out* 224*wer - 224*(wer-1)/2)
        # detections = detections.data.cpu()
        n_anchors = detections.shape[1]
        
        box_scales = torch.from_numpy(box_scales).unsqueeze(1).repeat(1,n_anchors)
        new_boxes = torch.from_numpy(new_boxes).unsqueeze(1).repeat(1,n_anchors,1)
        
        # add in original box offsets and scale outputs by original box scales
        detections[:,:,0] = detections[:,:,0]*box_scales/224 + new_boxes[:,:,1]
        detections[:,:,2] = detections[:,:,2]*box_scales/224 + new_boxes[:,:,1]
        detections[:,:,1] = detections[:,:,1]*box_scales/224 + new_boxes[:,:,2]
        detections[:,:,3] = detections[:,:,3]*box_scales/224 + new_boxes[:,:,2]

        # convert into xysr form 
        # output = np.zeros([len(detections),4])
        # output[:,0] = (detections[:,0] + detections[:,2]) / 2.0
        # output[:,1] = (detections[:,1] + detections[:,3]) / 2.0
        # output[:,2] = (detections[:,2] - detections[:,0])
        # output[:,3] = (detections[:,3] - detections[:,1]) / output[:,2]
        
        return detections
    
    
    def remove_overlaps(self):
        """
        Checks IoU between each set of tracklet objects and removes the newer tracklet
        when they overlap more than iou_cutoff (likely indicating a tracklet has drifted)
        """
        if self.iou_cutoff > 0:
            removals = []
            locations = self.filter.objs()
            for i in locations:
                for j in locations:
                    if i != j:
                        iou_metric = self.iou(locations[i],locations[j])
                        if iou_metric > self.iou_cutoff:
                            # determine which object has been around longer
                            if len(self.all_classes[i]) > len(self.all_classes[j]):
                                removals.append(j)
                            else:
                                removals.append(i)
            removals = list(set(removals))
            self.filter.remove(removals)
   
    
    def remove_anomalies(self,max_scale= 400):
        """
        Removes all objects with negative size or size greater than max_size
        """
        removals = []
        locations = self.filter.objs()
        for i in locations:
            if (locations[i][2]-locations[i][0]) > max_scale or (locations[i][2]-locations[i][0]) < 0:
                removals.append(i)
            elif (locations[i][3] - locations[i][1]) > max_scale or (locations [i][3] - locations[i][1]) < 0:
                removals.append(i)
        self.filter.remove(removals)         
    
    def iou(self,a,b):
        """
        Description
        -----------
        Calculates intersection over union for all sets of boxes in a and b
    
        Parameters
        ----------
        a : tensor of size [batch_size,4] 
            bounding boxes
        b : tensor of size [batch_size,4]
            bounding boxes.
    
        Returns
        -------
        iou - float between [0,1]
            average iou for a and b
        """
        
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        
        minx = max(a[0], b[0])
        maxx = min(a[2], b[2])
        miny = max(a[1], b[1])
        maxy = min(a[3], b[3])
        
        intersection = max(0, maxx-minx) * max(0,maxy-miny)
        union = area_a + area_b - intersection
        iou = intersection/union
        
        return iou

    def plot(self,im,detections,post_locations,all_classes,class_dict,frame = None):
        """
        Description
        -----------
        Plots the detections and the estimated locations of each object after 
        Kalman Filter update step
    
        Parameters
        ----------
        im : cv2 image
            The frame
        detections : tensor [n,4]
            Detections output by either localizer or detector (xysr form)
        post_locations : tensor [m,4] 
            Estimated object locations after update step (xysr form)
        all_classes : dict
            indexed by object id, where each entry is a list of the predicted class (int)
            for that object at every frame in which is was detected. The most common
            class is assumed to be the correct class        
        class_dict : dict
            indexed by class int, the string class names for each class
        frame : int, optional
            If not none, the resulting image will be saved with this frame number in file name.
            The default is None.
        """
        
        im = im.copy()/255.0
    
        # plot detection bboxes
        for det in detections:
            bbox = det[:4]
            color = (0.4,0.4,0.7) #colors[int(obj.cls)]
            c1 =  (int(bbox[0]),int(bbox[1]))
            c2 =  (int(bbox[2]),int(bbox[3]))
            cv2.rectangle(im,c1,c2,color,1)
            
        # plot estimated locations
        for id in post_locations:
            # get class
            try:
                most_common = np.argmax(all_classes[id])
                cls = class_dict[most_common]
            except:
                cls = "" 
            label = "{} {}".format(cls,id)
            bbox = post_locations[id][:4]
            
            if sum(bbox) != 0: # all 0's is the default in the storage array, so ignore these
                color = (0.7,0.7,0.4) #colors[int(obj.cls)]
                c1 =  (int(bbox[0]),int(bbox[1]))
                c2 =  (int(bbox[2]),int(bbox[3]))
                cv2.rectangle(im,c1,c2,color,1)
                
                # plot label
                text_size = 0.8
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN,text_size , 1)[0]
                c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(im, c1, c2,color, -1)
                cv2.putText(im, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,text_size, [225,255,255], 1);
        
        # resize to fit on standard monitor
        if im.shape[0] > 1920:
            im = cv2.resize(im, (1920,1080))
        cv2.imshow("window",im)
        cv2.waitKey(1)
        
        if self.writer is not None:
            self.writer(im)

    def parse_detections(self,scores,labels,boxes):
        """
        Description
        -----------
        Removes any duplicates from raw YOLO detections and converts from 8-D Yolo
        outputs to 6-d form needed for tracking
        
        input form --> batch_idx, xmin,ymin,xmax,ymax,objectness,max_class_conf, class_idx 
        output form --> x_center,y_center, scale, ratio, class_idx, max_class_conf
        
        Parameters
        ----------
        detections - tensor [n,8]
            raw YOLO-format object detections
        keep - list of int
            class indices to keep, default are vehicle class indexes (car, truck, motorcycle, bus)
        """
        if len(scores) == 0:
            return []
        
        # remove duplicates
        cutoff = torch.ones(scores.shape) * self.det_conf_cutoff
        keepers = torch.where(scores > cutoff)
        
        labels = labels[keepers]
        detections  = boxes[keepers]
        scores = scores[keepers]
        # input form --> batch_idx, xmin,ymin,xmax,ymax,objectness,max_class_conf, class_idx 
        # output form --> x_center,y_center, scale, ratio, class_idx, max_class_conf
        
        output = torch.zeros(detections.shape[0],6)
        output[:,0] = detections[:,0] 
        output[:,1] = detections[:,1] 
        output[:,2] = detections[:,2]
        output[:,3] = detections[:,3]
        output[:,4] =  labels
        output[:,5] =  scores
        
        
        return output

    def match_hungarian(self,first,second,dist_threshold = 50):
        """
        Description
        -----------
        performs  optimal (in terms of sum distance) matching of points 
        in first to second using the Hungarian algorithm
        
        inputs - N x 2 arrays of object x and y coordinates from different frames
        output - M x 1 array where index i corresponds to the second frame object 
            matched to the first frame object i
    
        Parameters
        ----------
        first - np.array [n,2]
            object x,y coordinates for first frame
        second - np.array [m,2]
            object x,y coordinates for second frame
        iou_cutoff - float in range[0,1]
            Intersection over union threshold below which match will not be considered
        
        Returns
        -------
        out_matchings - np.array [l]
            index i corresponds to second frame object matched to first frame object i
            l is not necessarily equal to either n or m (can have unmatched object from both frames)
        
        """
        # find distances between first and second
        if True:
            dist = np.zeros([len(first),len(second)])
            for i in range(0,len(first)):
                for j in range(0,len(second)):
                    dist[i,j] = np.sqrt((first[i,0]-second[j,0])**2 + (first[i,1]-second[j,1])**2)
        else:
            dist = np.zeros([len(first),len(second)])
            for i in range(0,len(first)):
                for j in range(0,len(second)):
                    dist[i,j] = 1 - self.iou(first[i],second[j].data.numpy())
                
        try:
            a, b = linear_sum_assignment(dist)
        except ValueError:
            print(dist,first,second)
            raise Exception
        
        # convert into expected form
        matchings = np.zeros(len(first))-1
        for idx in range(0,len(a)):
            matchings[a[idx]] = b[idx]
        matchings = np.ndarray.astype(matchings,int)
        
        # remove any matches too far away
        for i in range(len(matchings)):
            try:
                if dist[i,matchings[i]] > dist_threshold:
                    matchings[i] = -1
            except:
                pass
            
        # write into final form
        out_matchings = []
        for i in range(len(matchings)):
            if matchings[i] != -1:
                out_matchings.append([i,matchings[i]])
        return np.array(out_matchings)
        
    def md_iou(self,a,b):
        """
        a,b - [batch_size x num_anchors x 4]
        """
        
        area_a = (a[:,:,2]-a[:,:,0]) * (a[:,:,3]-a[:,:,1])
        area_b = (b[:,:,2]-b[:,:,0]) * (b[:,:,3]-b[:,:,1])
        
        minx = torch.max(a[:,:,0], b[:,:,0])
        maxx = torch.min(a[:,:,2], b[:,:,2])
        miny = torch.max(a[:,:,1], b[:,:,1])
        maxy = torch.min(a[:,:,3], b[:,:,3])
        zeros = torch.zeros(minx.shape,dtype=float)
        
        intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
        union = area_a + area_b - intersection
        iou = torch.div(intersection,union)
        
        #print("MD iou: {}".format(iou.max(dim = 1)[0].mean()))
        return iou

    def track(self):
        """
        Returns
        -------
        final_output : list of lists, one per frame
            Each sublist contains dicts, one per estimated object location, with fields
            "bbox", "id", and "class_num"
        frame_rate : float
            number of frames divided by total processing time
        time_metrics : dict
            Time utilization for each operation in tracking
        """    
        self.start_time = time.time()
        
        
        frame_stuff = next(self.loader)            
        if len(frame_stuff) == 5:
            (frame_num,frame,dim,original_im,timestamp) = frame_stuff
            self.all_timestamps.append(timestamp[0])
        else:
            (frame_num,frame,dim,original_im) = frame_stuff
            self.all_timestamps.append(-1)
            
        while frame_num != -1:            
            
            #if frame_num % self.d < self.init_frames:
            # predict next object locations
            start = time.time()
            try: # in the case that there are no active objects will throw exception
                self.filter.predict()
                pre_locations = self.filter.objs()
            except:
                pre_locations = []    
                
            pre_ids = []
            pre_loc = []
            for id in pre_locations:
                pre_ids.append(id)
                pre_loc.append(pre_locations[id])
            pre_loc = np.array(pre_loc)  
            
            self.time_metrics['predict'] += time.time() - start
        
            # check1 = len(pre_loc)
            # try:
            #     check2 = torch.sqrt(torch.sum(self.filter.X[:,4]) + torch.sum(self.filter.X[:,5]))
            # except:
            #     pass
            
            if frame_num % self.d < self.init_frames:  
                
                # detection step
                try: # use CNN detector
                    start = time.time()
                    with torch.no_grad():                       
                        scores,labels,boxes = self.detector(frame.unsqueeze(0))            
                        torch.cuda.synchronize(self.device)
                    self.time_metrics['detect'] += time.time() - start
                    
                    # move detections to CPU
                    start = time.time()
                    scores = scores.cpu()
                    labels = labels.cpu()
                    boxes = boxes.cpu()
                    self.time_metrics['load'] += time.time() - start
                
                except: # use mock detector
                    scores,labels,boxes,time_taken = self.detector(self.track_id,frame_num)
                    self.time_metrics["detect"] += time_taken
                    
               
    
                # postprocess detections
                start = time.time()
                detections = self.parse_detections(scores,labels,boxes)
                self.time_metrics['parse'] += time.time() - start
             
                # match using Hungarian Algorithm        
                start = time.time()
                # matchings[i] = [a,b] where a is index of pre_loc and b is index of detection
                matchings = self.match_hungarian(pre_loc,detections,dist_threshold = self.matching_cutoff)
                self.time_metrics['match'] += time.time() - start
                
                # Update tracked objects
                self.manage_tracks(detections,matchings,pre_ids)
                
            # skip  if there are no active tracklets or no localizer (KF prediction with no update)    
            elif len(pre_locations) > 0 and self.localizer is not None and (frame_num % self.d)%self.s == 0:
                
                # get crop for each active tracklet
                crops,new_boxes,box_ids,box_scales = self.crop_tracklets(pre_locations,frame)
                
                
                # localize objects using localizer
                start= time.time()
                with torch.no_grad():                       
                     reg_boxes, classes = self.localizer(crops,LOCALIZE = True)


                    
                del crops
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.time_metrics['localize'] += time.time() - start
                start = time.time()
                
               
                # convert to global coords
                
                reg_boxes = reg_boxes.data.cpu()
                classes = classes.data.cpu()
                self.time_metrics["load"] += time.time() - start
                start = time.time()

                reg_boxes = self.local_to_global(reg_boxes,box_scales,new_boxes)
                
                # parse retinanet detections
                confs,classes = torch.max(classes, dim = 2)
                
                # use original bboxes to weight best bboxes 
                n_anchors = reg_boxes.shape[1]
                a_priori = torch.from_numpy(pre_loc[:,:4])
                # bs = torch.from_numpy(box_scales).unsqueeze(1).repeat(1,4)
                # a_priori = a_priori * 224/bs
                a_priori = a_priori.unsqueeze(1).repeat(1,n_anchors,1)
                
                iou_score = self.md_iou(a_priori.double(),reg_boxes.double())
                score = confs + iou_score
                best_scores ,keep = torch.max(score,dim = 1)
                
                idx = torch.arange(reg_boxes.shape[0])
                detections = reg_boxes[idx,keep,:]
                cls_preds = classes[idx,keep]
                
                self.time_metrics["post_localize"] += time.time() -start
                start = time.time()
                
                # store class predictions
                for i in range(len(cls_preds)):
                    self.all_classes[box_ids[i]][cls_preds[i].item()] += 1


                
                # map regressed bboxes directly to objects for update step
                self.filter.update(detections,box_ids)
                self.time_metrics['update'] += time.time() - start
                
                # increment all fslds
                for i in range(len(pre_ids)):
                        self.fsld[pre_ids[i]] += 1
        
                # remove overlapping objects and anomalies
                self.remove_overlaps()
                self.remove_anomalies()
                
                
            # get all object locations and store in output dict
            start = time.time()
            try:
                post_locations = self.filter.objs()
            except:
                post_locations = {}
            for id in post_locations:
                try:
                   self.all_tracks[id][frame_num,:] = post_locations[id][:self.state_size]   
                except IndexError:
                    print("Index Error")
            self.time_metrics['store'] += time.time() - start  
            
            
            # 10. Plot
            start = time.time()
            if self.PLOT:
                self.plot(original_im,detections,post_locations,self.all_classes,self.class_dict,frame = frame_num)
            self.time_metrics['plot'] += time.time() - start
       
            # load next frame  
            self.frames_processed += 1
            start = time.time()
            
            frame_stuff = next(self.loader)            
            if len(frame_stuff) == 5:
                (frame_num,frame,dim,original_im,timestamp) = frame_stuff
                self.all_timestamps.append(timestamp[0])
            else:
                (frame_num,frame,dim,original_im) = frame_stuff
                self.all_timestamps.append(-1)

            print(frame_num)
            
            torch.cuda.synchronize()
            self.time_metrics["load"] = time.time() - start
            torch.cuda.empty_cache()
            
        
        # clean up at the end
        self.end_time = time.time()
        
        cv2.destroyAllWindows()
        
    def get_results(self):
        """
        Call after calling track to summarize results
        Returns:
            final_output - list of object tracklet locations per frame
            framerate - calculated fps
            time_metrics - time spent on each operation
        """
        if len(self.all_tracks) == 0:
            print("Must call track() before getting results")
            return None
        
        else:   
            # get speed results
            total_time = 0
            for key in self.time_metrics:
                total_time += self.time_metrics[key]
            #framerate = self.frames_processed / total_time
            framerate = self.frames_processed / (self.end_time - self.start_time())

        # get tracklet results 
        final_output = []
                    
        for frame in range(self.n_frames):
            frame_objs = []
            
            for id in self.all_tracks:
                bbox = self.all_tracks[id][frame]
                if bbox[0] != 0:
                    obj_dict = {}
                    obj_dict["id"] = id
                    obj_dict["class_num"] = np.argmax(self.all_classes[id])
                    x0 = bbox[0] #- bbox[2]/2.0
                    x1 = bbox[2] #+ bbox[2]/2.0
                    y0 = bbox[1] #- bbox[2]*bbox[3]/2.0
                    y1 = bbox[3] #+ bbox[2]*bbox[3]/2.0
                    obj_dict["bbox"] = np.array([x0,y0,x1,y1])
                    
                    frame_objs.append(obj_dict)
            final_output.append(frame_objs)
        
        return final_output, framerate, self.time_metrics
    
    def get_results_csv(self):
        """
        Call after tracking to summarize results in .csv file
        """
        outfile = self.input_file_name.split(".")[0] + "_track_outputs.csv"
        if self.output_dir is not None:
            outfile = os.path.join(self.output_dir,outfile.split("/")[-1])

        
        # create headers
        summary_header = [
            "Video sequence name",
            "Processing start time",
            "Processing end time",
            "Timestamp start time",
            "Timestamp end time",
            "Frames processed",
            "Unique objects",
            "GPU"
            ]
        
        # create header data
        summary = []
        summary.append(self.input_file_name)
        start_time_ms = str(np.round(self.start_time%1,2)).split(".")[1]
        start_time = time.strftime('%Y-%m-%d %H:%M:%S.{}', time.localtime(self.start_time)).format(start_time_ms)
        summary.append(start_time)
        end_time_ms = str(np.round(self.end_time%1,2)).split(".")[1]
        end_time = time.strftime('%Y-%m-%d %H:%M:%S.{}', time.localtime(self.end_time)).format(end_time_ms)
        summary.append(end_time)    
        summary.append(self.all_timestamps[0])
        summary.append(self.all_timestamps[-1])
        summary.append(self.frames_processed)
        summary.append(self.next_obj_id) # gives correct count since 0-indexed
        summary.append(str(self.device))
        
        fps = self.frames_processed / (self.end_time - self.start_time)
        time_header = ["Processing fps"]
        time_data = [fps]
        for item in self.time_metrics.keys():
            time_header.append(item)
            time_data.append(self.time_metrics[item])
        
        data_header = [
            "Frame #",
            "Timestamp",
            "Object ID",
            "Object Class",
            "BBox xmin",
            "BBox ymin",
            "BBox xmax",
            "BBox ymax"
            ]
        
        with open(outfile, mode='w') as f:
            out = csv.writer(f, delimiter=',')
            
            # write first chunk
            out.writerow(summary_header)
            out.writerow(summary)
            out.writerow([])
            
            # write second chunk
            out.writerow(time_header)
            out.writerow(time_data)
            out.writerow([])
            
            # write main chunk
            out.writerow(data_header)
            
            for frame in range(self.frames_processed):
                timestamp = self.all_timestamps[frame]
                
                for id in self.all_tracks:
                    bbox = self.all_tracks[id][frame]
                    if bbox[0] != 0:
                        obj_line = []
                        obj_line.append(frame)
                        obj_line.append(timestamp)
                        obj_line.append(id)
                        obj_line.append(self.class_dict[np.argmax(self.all_classes[id])])
                        obj_line.append(bbox[0])
                        obj_line.append(bbox[2])
                        obj_line.append(bbox[1])
                        obj_line.append(bbox[3])
                        
                        out.writerow(obj_line)
            
            
            
        