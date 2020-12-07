#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:33:03 2020

@author: worklab
"""


import os,sys
import numpy as np
import random 
import time
random.seed = 0

import cv2
from PIL import Image
import torch

from torchvision.transforms import functional as F
import torch.multiprocessing as mp

# import timestamp parsing
try:
    cwd = os.getcwd()
    gp_cwd = cwd.split("/")[:-2]
    gp_cwd.append("I24-video-ingest")
    ts_path = os.path.join("/",*gp_cwd)
    if ts_path not in sys.path:
        sys.path.insert(0,ts_path)
    
    from utilities import get_precomputed_checksums,get_timestamp_geometry,parse_frame_timestamp
    
except:
    cwd = os.getcwd()
    gp_cwd = cwd.split("/")[:-1]
    gp_cwd.append("I24-video-ingest")
    ts_path = os.path.join("/",*gp_cwd)
    if ts_path not in sys.path:
        sys.path.insert(0,ts_path)
    
    from utilities import get_precomputed_checksums,get_timestamp_geometry,parse_frame_timestamp
class FrameLoader():
    
    def __init__(self,track_directory,device, buffer_size = 9,timestamp_geom_path = None,timestamp_checksum_path = None):
        
        """
        Parameters
        ----------
        track_directory : str
            Path to frames 
        device : torch.device
            specifies where frames should be loaded to , memory or GPU
        det_step : int
            How often full detection is performed, necessary for image pre-processing
        init_frames : int
            Number of frames used to initialize Kalman filter every det_step frames
        cutoff : int, optional
            If not None, only this many frames are loaded into memory. The default is None.
    
        """
        try:
            files = []
            for item in [os.path.join(track_directory,im) for im in os.listdir(track_directory)]:
                files.append(item)
                files.sort()    
        
            self.files = files
            
            manager = mp.Manager()
            
            #self.det_step = det_step
            self.device = device
        
            # create shared queue
            #mp.set_start_method('spawn')
            ctx = mp.get_context('spawn')
            self.queue = ctx.Queue()
            
            self.frame_idx = -1
            
            self.worker = ctx.Process(target=load_to_queue, args=(self.queue,files,device,buffer_size,))
            self.worker.start()
            time.sleep(5)
        
        except: # file is a video
            sequence = track_directory
            
            self.sequence = sequence
        
            manager = mp.Manager()
            
            #self.det_step = det_step
            self.device = device
        
            # create shared queue
            #mp.set_start_method('spawn')
            ctx = mp.get_context('spawn')
            self.queue = ctx.Queue()
            
            self.frame_idx = -1
            
            args=(self.queue,sequence,device,buffer_size,)
            kwargs = {"checksum_path":timestamp_checksum_path,"geom_path":timestamp_geom_path}
            self.worker = ctx.Process(target=load_to_queue_video,args = args, kwargs=kwargs)
            self.worker.start()
            time.sleep(5)
        
    def __len__(self):
        """
        Description
        -----------
        Returns number of frames in the track directory
        """
        try:
            return self.len
        except:   
            return len(self.files)
        finally:
            return None
    
    def __next__(self):
        """
        Description
        -----------
        Returns next frame and associated data unless at end of track, in which
        case returns -1 for frame num and None for frame

        Returns
        -------
        frame_num : int
            Frame index in track
        frame : tuple of (tensor,tensor,tensor)
            image, image dimensions and original image

        """
        frame = self.queue.get(timeout = 10)
        self.frame_idx = frame[0]
        
        if self.frame_idx != -1:
            return frame
        
        else:
            self.worker.terminate()
            self.worker.join()
            return frame

def load_to_queue(image_queue,files,device,queue_size):
    """
    Description
    -----------
    Whenever necessary, loads images, moves them to GPU, and adds them to a shared
    multiprocessing queue with the goal of the queue always having a certain size.
    Process is to be called as a worker by FrameLoader object
    
    Parameters
    ----------
    image_queue : multiprocessing Queue
        shared queue in which preprocessed images are put.
    files : list of str
        each str is path to one file in track directory
    init_frames : int
        specifies number of dense detections before localization begins
    device : torch.device
        Specifies whether images should be put on CPU or GPU.
    queue_size : int, optional
        Goal size of queue, whenever actual size is less additional images will
        be processed and added. The default is 5.
    """
    
    frame_idx = 0    
    while frame_idx < len(files):
        
        if image_queue.qsize() < queue_size:
            
            # load next image
            with Image.open(files[frame_idx]) as im:
             
              # if frame_idx % det_step.value < init_frames:   
              #     # convert to CV2 style image
              #     open_cv_image = np.array(im) 
              #     im = open_cv_image.copy() 
              #     original_im = im[:,:,[2,1,0]].copy()
              #     # new stuff
              #     dim = (im.shape[1], im.shape[0])
              #     im = cv2.resize(im, (1920,1080))
              #     im = im.transpose((2,0,1)).copy()
              #     im = torch.from_numpy(im).float().div(255.0).unsqueeze(0)
              #     dim = torch.FloatTensor(dim).repeat(1,2)
              #     dim = dim.to(device,non_blocking = True)
              # else:
                  # keep as tensor
              original_im = np.array(im)[:,:,[2,1,0]].copy()
              im = F.to_tensor(im)
              im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
              dim = None
                 
              # store preprocessed image, dimensions and original image
              im = im.to(device)
              frame = (frame_idx,im,dim,original_im)
             
              # append to queue
              image_queue.put(frame)
             
            frame_idx += 1
    
    image_queue.put((-1,None,None,None))
    
    # neverending loop, because if the process ends, the tensors originally
    # initialized in this function will be deleted, causing issues. Thus, this 
    # function runs until a call to self.next() returns -1, indicating end of track 
    # has been reached
    while True:  
           time.sleep(5)
        
def load_to_queue_video(image_queue,sequence,device,queue_size,checksum_path = None,geom_path = None):
    
    cap = cv2.VideoCapture(sequence)
    
    if checksum_path is not None:
        checksums = get_precomputed_checksums(checksum_path)
        geom = get_timestamp_geometry(geom_path)
    
    frame_idx = 0    
    while True:
        
        if image_queue.qsize() < queue_size:
            
            
            
            # load next image from videocapture object
            ret,original_im = cap.read()
            if ret == False:
                frame = (-1,None,None,None,None)
                image_queue.put(frame)       
                break
            else:
                timestamp = None
                if checksum_path is not None:
                    # get timestamp
                    timestamp = parse_frame_timestamp(frame_pixels = original_im, timestamp_geometry = geom, precomputed_checksums = checksums)
                    if timestamp[0] is None:
                        #print(original_im.shape)
                        #cv2.imshow("frame",timestamp[1])
                        #cv2.waitKey(0)
                        timestamp = None
                    
                original_im = cv2.resize(original_im,(1920,1080))
                im = F.to_tensor(original_im)
                im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                # store preprocessed image, dimensions and original image
                im = im.to(device)
                dim = None
                frame = (frame_idx,im,dim,original_im,timestamp)
             
                # append to queue
                image_queue.put(frame)       
                frame_idx += 1
    
    # neverending loop, because if the process ends, the tensors originally
    # initialized in this function will be deleted, causing issues. Thus, this 
    # function runs until a call to self.next() returns -1, indicating end of track 
    # has been reached
    while True:  
           time.sleep(5)
    
    
def test_frameloader(path,geom,checksum):
    
    test = FrameLoader(path,torch.device("cuda:3"),buffer_size = 15,timestamp_geom_path = geom,timestamp_checksum_path = checksum)
    
    print (test.queue.qsize())
    all_time = 0
    count = 0
    
    while True:
        start = time.time()
        (frame_idx,im,dim,original_im,timestamp) = next(test)
        
        if frame_idx > 0:
            all_time += (time.time() - start)
        
        time.sleep(0.03)
       
        # try modifying the frame
        if im is not None:
            out = im[0] + 1
                
        if frame_idx == -1:
            break
        
        count += 1
        print("Frame {}, timestamp: {},  Loading rate:{}   Queue size:{}".format(count, timestamp, all_time/count,test.queue.qsize()))
        
def test_load_to_queue_video(sequence,queue_size,checksum_path = None,geom_path = None):
        import queue
        test_queue = queue.Queue(queue_size)
        
        load_to_queue_video(test_queue,sequence,torch.device("cuda:3"),queue_size,checksum_path = checksum_path, geom_path = geom_path)
        
        
if __name__ == "__main__":
    
    path = "/home/worklab/Data/cv/video/ingest_session_00011/recording/record_p1c0_00000.mp4"
    geom = "/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_geometry_4K.pkl"
    checksum = "/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_pixel_checksum_6.pkl"
    #test_load_to_queue_video(path,10,geom_path = geom,checksum_path = checksum)
    test_frameloader(path,geom,checksum)