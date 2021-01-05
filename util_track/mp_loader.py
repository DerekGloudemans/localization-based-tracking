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
import queue
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
    try:
        cwd = os.getcwd()
        gp_cwd = cwd.split("/")[:-1]
        gp_cwd.append("I24-video-ingest")
        ts_path = os.path.join("/",*gp_cwd)
        if ts_path not in sys.path:
            sys.path.insert(0,ts_path)
        from utilities import get_precomputed_checksums,get_timestamp_geometry,parse_frame_timestamp
    except:
        sys.path.insert(0,os.path.join(os.getcwd(),"I24-video-ingest"))
        from utilities import get_precomputed_checksums,get_timestamp_geometry,parse_frame_timestamp
        
class FrameLoader():
    
    def __init__(self,track_directory,device, buffer_size = 15,timestamp_geom_path = None,timestamp_checksum_path = None,com_queue = None,s=1,d=1):
        
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
            self.timeout = buffer_size
            self.com_queue = com_queue
            
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
            self.timeout = buffer_size
            self.com_queue = com_queue
            
            self.frame_idx = -1
            
            args=(self.queue,sequence,device,buffer_size,)
            kwargs = {"checksum_path":timestamp_checksum_path,"geom_path":timestamp_geom_path,"com_queue":com_queue,"s":s,"d":d}
            self.worker = ctx.Process(target=load_to_queue_video,args = args, kwargs=kwargs,daemon = True)
            self.worker.start()
            time.sleep(buffer_size)
        
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
        while True:
            try:
                frame = self.queue.get(timeout = self.timeout)
                self.frame_idx = frame[0]
                break
            except queue.Empty:
                worker_id = int(str(self.device).split(":")[1])
                ts = time.time()
                key = "WARNING"
                message = "Loader {} main (PID {}) timed out __next__. Sleeping for 2 seconds.".format(worker_id,os.getpid()) 
                self.com_queue.put((ts,key,message,worker_id))
                time.sleep(2)

            
        if self.frame_idx != -1:
            return frame
        
        else:
            if self.com_queue is not None:
                worker_id = int(str(self.device).split(":")[1])
                ts = time.time()
                key = "INFO"
                message = "Loader {} main (PID {}) terminating worker.".format(worker_id,os.getpid()) 
                self.com_queue.put((ts,key,message,worker_id))
            
            self.worker.terminate()
            self.worker.join()
            
            if self.com_queue is not None:
                worker_id = int(str(self.device).split(":")[1])
                ts = time.time()
                key = "INFO"
                message = "Loader {} main (PID {}) successfully terminated worker.".format(worker_id,os.getpid()) 
                self.com_queue.put((ts,key,message,worker_id))
            
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
              #im = F.resize(im,(1920,1080)) # downsample to 1080p
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
        
def load_to_queue_video(image_queue,sequence,device,queue_size,s=1,d=1,checksum_path = None,geom_path = None,com_queue = None):
    
    cap = cv2.VideoCapture(sequence)
    
    if checksum_path is not None:
        checksums = get_precomputed_checksums(checksum_path)
        geom = get_timestamp_geometry(geom_path)
    
    worker_id = int(str(device).split(":")[1])
    
    if com_queue is not None:
        ts = time.time()
        key = "INFO"
        message = "Loader {} worker (PID {}) initialized successfully on sequence {}.".format(worker_id,os.getpid(),sequence)
        com_queue.put((ts,key,message,worker_id))
    
    time_metrics = {"decode":1e-04,
                    "timestamp":1e-04,
                    "tensor":1e-04,
                    "preprocess":1e-04,
                    "transfer":1e-04,
                    "queue":1e-04}
    
    frame_idx = 0    
    while True:
        
        if image_queue.qsize() < queue_size:
            
            
            if com_queue is not None and frame_idx % 100 == 0:
                total = sum([time_metrics[key] for key in time_metrics])
                tutil = []
                for key in time_metrics:
                    tutil.append("{}:{}s ({}%)".format(key,round(time_metrics[key]/(frame_idx+1e-04),4),round(time_metrics[key]/total*100,1)))
                message = "Loader {} worker (PID {}) time utilization: {}".format(worker_id,os.getpid(),tutil)
                ts = time.time()
                key = "INFO"
                com_queue.put((ts,key,message,worker_id))

            try:
            
                # load next image from videocapture object
                start = time.time()
                ret_grab = cap.grab()
                
                if  ret_grab and (frame_idx %d == 0 or (frame_idx % d)%s == 0): # only decode the frame if we have to
                    ret, original_im = cap.retrieve()
                
                    time_metrics["decode"] += time.time() - start
                    
                    if ret == False:
                        frame = (-1,None,None,None,None)
                        image_queue.put(frame)
                        
                        if com_queue is not None:
                            ts = time.time()
                            key = "DEBUG"
                            message = "Loader {} worker (PID {}) sent last frame ({} total sent) for sequence.".format(worker_id,os.getpid(),frame_idx)
                            com_queue.put((ts,key,message,worker_id))
                            
                        break
                    
                    else:
                        start = time.time()
                        timestamp = None
                        if checksum_path is not None:
                            # get timestamp
                            timestamp = parse_frame_timestamp(frame_pixels = original_im, timestamp_geometry = geom, precomputed_checksums = checksums)
                            if timestamp[0] is None:
                                timestamp = None
                        time_metrics["timestamp"] += time.time() - start
    
                        start = time.time()
                        #original_im = cv2.resize(original_im,(1920,1080))
                        im = F.to_tensor(original_im)
                        time_metrics["tensor"] += time.time() - start
    
                        #if worker_id <2 and frame_idx <2:
                            #cv2.imwrite("/isis/code/I24-video-processing/snapshots/{}_{}.png".format(worker_id,frame_idx),original_im)
                            #cv2.imwrite("/home/worklab/Documents/derek/I24-video-processing/snapshots/{}_{}.png".format(worker_id,frame_idx),original_im)

                        start = time.time()
                        im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                        time_metrics["preprocess"] += time.time() - start
    
                        # store preprocessed image, dimensions and original image
                        start = time.time()
                        im = im.to(device,non_blocking=True) # change back if this causes errors
                        dim = None
                        original_im = None # need to change this so that it passes the original image if
                        frame = (frame_idx,im,dim,original_im,timestamp)
                        time_metrics["transfer"] += time.time() - start
    
                        # append to queue
                        start = time.time()
                        image_queue.put(frame)       
                        frame_idx += 1
                        time_metrics["queue"] += time.time() - start
                        
                else: # otherwise, pass a dummy value if the frame won't be used anyway
                    time_metrics["decode"] += time.time() - start
                    
                    if ret_grab == False:
                        frame = (-1,None,None,None,None)
                        image_queue.put(frame)
                        
                        if com_queue is not None:
                            ts = time.time()
                            key = "DEBUG"
                            message = "Loader {} worker (PID {}) sent last frame (skipped, {} total sent) for sequence.".format(worker_id,os.getpid(),frame_idx)
                            com_queue.put((ts,key,message,worker_id))
                            
                        break
                    
                    else:
                        frame = (frame_idx,None,None,None,None)
                        image_queue.put(frame)
                        frame_idx +=1

            
            except Exception as e:
                if com_queue is not None:
                    ts = time.time()
                    key = "ERROR"
                    message = "Loader {} worker (PID {}) error: .".format(worker_id,os.getpid(),e)
                    com_queue.put((ts,key,message,worker_id))
        
    # neverending loop, because if the process ends, the tensors originally
    # initialized in this function will be deleted, causing issues. Thus, this 
    # function runs until a call to self.next() returns -1, indicating end of track 
    # has been reached
    if com_queue is not None:
        ts = time.time()
        key = "DEBUG"
        message = "Loader {} worker (PID {}) waiting for loader to terminate worker process".format(worker_id,os.getpid())
        com_queue.put((ts,key,message,worker_id))
    while True:  
        time.sleep(5)
    
    
def test_frameloader(path,geom,checksum):
    
    test = FrameLoader(path,torch.device("cuda:3"),buffer_size = 15,timestamp_geom_path = geom,timestamp_checksum_path = checksum,d =1,s=1)
    
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