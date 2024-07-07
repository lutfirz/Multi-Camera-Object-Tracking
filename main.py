import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import time

from LoFTR.src.utils.plotting import make_matching_figure
from LoFTR.src.loftr import LoFTR, default_cfg

import sort
import utilities
import homography_tracker

video1 = cv2.VideoCapture("path")
video2 = cv2.VideoCapture("path")  

_, frame1 = video1.read()
_, frame2 = video2.read()

cv2.imwrite("frame1.jpg", frame1)
cv2.imwrite("frame2.jpg", frame2)
image_pair = ["frame1.jpg", "frame2.jpg"]

matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("model.ckpt")['state_dict'])
matcher = matcher.eval().cpu()

img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (1280, 720))
img1_raw = cv2.resize(img1_raw, (1280, 720))

img0 = torch.from_numpy(img0_raw)[None][None].cpu() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cpu() / 255.
batch = {'image0': img0, 'image1': img1}

with torch.no_grad():
  matcher(batch)
  mkpts0 = batch['mkpts0_f'].cpu().numpy()
  mkpts1 = batch['mkpts1_f'].cpu().numpy()
  mconf = batch['mconf'].cpu().numpy()

confidence_threshold = 0.25
high_conf_indices = mconf >= confidence_threshold
mkpts0 = mkpts0[high_conf_indices]
mkpts1 = mkpts1[high_conf_indices]
mconf = mconf[high_conf_indices]

color = cm.jet(mconf, alpha=0.7)
text = [
  'LoFTR',
  'Matches: {}'.format(len(mkpts0)),
]

make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text, path="LoFTR")

cam4_H_cam1, status = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
cam1_H_cam4 = np.linalg.inv(cam4_H_cam1)

homographies = list()
homographies.append(np.eye(3))
homographies.append(cam1_H_cam4)

detector = torch.hub.load("ultralytics/yolov5", "yolov5m")
detector.agnostic = True
detector.classes = [0]
detector.conf = 0.3

trackers = [
  sort.Sort(
    max_age=30, 
    min_hits=3, 
    iou_threshold=0.3
  )
  for _ in range(2)
]

global_tracker = homography_tracker.MultiCameraTracker(homographies, iou_thres=0.20)

#video1.set(cv2.CAP_PROP_POS_FRAMES, x)
#video2.set(cv2.CAP_PROP_POS_FRAMES, y)

while True:   
  frame1 = video1.read()[1]
  frame2 = video2.read()[1]

  frame1 = cv2.resize(frame1, (1280, 720))
  frame2 = cv2.resize(frame2, (1280, 720))

  frames = [frame1[:, :, ::-1], frame2[:, :, ::-1]]

  anno = detector(frames)

  dets, tracks = [], []

  for i in range(len(anno)):
    det = anno.xyxy[i].cpu().numpy()
    det[:, :4] = np.intp(det[:, :4])
    dets.append(det)

    tracker = trackers[i].update(det[:, :4], det[:, -1])
    tracks.append(tracker)

  global_ids = global_tracker.update(tracks)

  for i in range(2):
    frames[i] = utilities.draw_tracks(
      frames[i][:, :, ::-1],
      tracks[i],
      global_ids[i],
      i,
      classes=detector.names,
    )

  vis = np.hstack(frames)

  org1 = (30, 90)  
  font = cv2.FONT_HERSHEY_SIMPLEX 
  fontScale = 3  
  color = (0, 255, 0) 
  thickness = 7  

  vis = cv2.putText(vis, "Detected: " + str(len(det)), org1, font, fontScale, color, thickness)

  cv2.namedWindow("Vis", cv2.WINDOW_NORMAL)
  cv2.imshow("Vis", vis)
  
  key = cv2.waitKey(1)
  if key == ord("q"):
    break

video1.release()
video2.release()
cv2.destroyAllWindows()