import os,sys,inspect
import numpy as np
import random 
import time
import math
import _pickle as pickle
random.seed = 0

import cv2
from PIL import Image
import torch

import matplotlib.pyplot  as plt

# tracker
from tracker import Localization_Tracker

# CNNs
detector_path = os.path.join(os.getcwd(),"models","pytorch_retinanet_detector")
sys.path.insert(0,detector_path)
from models.pytorch_retinanet_detector.retinanet.model import resnet50 

localizer_path = os.path.join(os.getcwd(),"models","pytorch_retinanet_localizer")
sys.path.insert(0,localizer_path)
from models.pytorch_retinanet_localizer.retinanet.model import resnet34

detrac_util_path = os.path.join(os.getcwd(),"util_detrac")
sys.path.insert(0,detrac_util_path)
from util_detrac.detrac_detection_dataset import class_dict









if __name__ == "__main__":

     # input directory
     track_dir = os.path.join(os.getcwd(),"demo")     
    
     # parameters
     iou_cutoff = 0.75       # tracklets overlapping by this amount will be pruned
     det_step = 10            # frames between full detection steps
     skip_step = 1           # frames between update steps (either detection or localization)
     ber = 2.4               # amount by which to expand a priori tracklet to crop object
     init_frames = 1         # number of detection steps in a row to perform
     matching_cutoff = 100   # detections farther from tracklets than this will not be matched 
     SHOW = True             # show tracking in progress?
     LOCALIZE = True        # if False, will skip frames between detection steps
     OUTVID = os.path.join(os.getcwd(),"demo","example_outputs")
    
    ###########################################################################
    
     # enable CUDA
     use_cuda = torch.cuda.is_available()
     device_id = 0
     device = torch.device("cuda:{}".format(device_id) if use_cuda else "cpu")
    
     # load localizer
     loc_cp = os.path.join(os.getcwd(),"config","localizer_state_dict.pt")
     localizer = resnet34(13,device_id = device_id)
     localizer.load_state_dict(torch.load(loc_cp))
     localizer = localizer.to(device)
     localizer.eval()
     localizer.training = False    
     
     # load detector
     det_cp = os.path.join(os.getcwd(),"config","detector_state_dict.pt")
     det_cp = os.path.join(os.getcwd(),"config","i24_detector_4k_state_dict_e69.pt")
     detector = resnet50(9,device_id = device_id)
     detector.load_state_dict(torch.load(det_cp))
     # detector = detector.to(device)
     # detector.eval()
     # detector.training = False   
     
     # load filter
     filter_params = os.path.join(os.getcwd(),"config","filter_params_tuned.cpkl")
     
     # for timestamps, remove later
     geom = "/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_geometry_4K.pkl"
     checksum = "/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_pixel_checksum_6.pkl"
     
     if not LOCALIZE:
         localizer = None

     with open(filter_params ,"rb") as f:
                 kf_params = pickle.load(f)
                 # these adjustments make the filter a bit less laggy
                 kf_params["R"] /= 20
                 kf_params["R2"] /= 50 
                                         
     # get all sequences
     for sequence in os.listdir(track_dir):
        if sequence == "example_outputs" or ".cpkl" in sequence:
            continue
        
        sequence_path = os.path.join(track_dir,sequence)
        
        # track it!
        tracker = Localization_Tracker(sequence_path,
                                       detector,
                                       localizer,
                                       kf_params,
                                       class_dict,
                                       device_id = device_id,
                                       det_step = det_step,
                                       skip_step = 1,
                                       init_frames = init_frames,
                                       fsld_max = det_step*2,
                                       det_conf_cutoff = 0.5,
                                       matching_cutoff = matching_cutoff,
                                       iou_cutoff = iou_cutoff,
                                       ber = ber,
                                       PLOT = SHOW,
                                       OUT = OUTVID,
                                       wer = 1.25,
                                       checksum_path = checksum,
                                       geom_path = geom)
        
        tracker.track()
        preds, Hz, time_metrics = tracker.get_results()
        tracker.write_results_csv()
        
        print("Finished sequence {}, tracked at {} fps".format(sequence,Hz))
        if SHOW:
            print("For a more accurate fps estimate, turn plotting off")
            
        save_file = "{}_results_d{}.cpkl".format(sequence,det_step)
        with open(save_file,"wb") as f:
            pickle.dump(preds,f)
                    

                            
               