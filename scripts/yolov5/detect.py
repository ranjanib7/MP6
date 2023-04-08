# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
#if str(ROOT) not in sys.path:
    #sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#sys.path.insert(1, os.getcwd())'''

MIN_POINTS = 50
mtx = []
dist = []
tvecs = []

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import smart_inference_mode, select_device
import imutils

def calibrate_cam():
    # calibration
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ## setup simpleblob detector parameters
    blobParams = cv2.SimpleBlobDetector_Params()

    # change thresholds
    blobParams.minThreshold = 8
    blobParams.maxThreshold = 255

    # filter by area
    blobParams.filterByArea = True
    blobParams.minArea = 64     # minArea may be adjusted to suit for your experiment
    blobParams.maxArea = 2500   # maxArea may be adjusted to suit for your experiment

    # Filter by Circularity
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.1

    # Filter by Convexity
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.87

    # Filter by Inertia
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01

    # Create a detector with the parameters
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)

    ###################################################################################################

    # Original blob coordinates, supposing all blobs are of z-coordinates 0
    # And, the distance between every two neighbour blob circle centers is 3.175 centimetres/inches?
    objp = np.zeros((44, 3), np.float32)

    objp[0]  = (0  , 0  , 0)
    objp[1]  = (0  , 3.175 , 0)
    objp[2]  = (0  , 6.35, 0)
    objp[3]  = (0  , 9.525, 0)
    objp[4]  = (1.5875 , 1.5875 , 0)
    objp[5]  = (1.5875 , 4.7625, 0)
    objp[6]  = (1.5875 , 7.9375, 0)
    objp[7]  = (1.5875 , 11.1125, 0)
    objp[8]  = (3.175 , 0  , 0)
    objp[9]  = (3.175 , 3.175 , 0)
    objp[10] = (3.175 , 6.35, 0)
    objp[11] = (3.175 , 9.525, 0)
    objp[12] = (4.7625, 1.5875,  0)
    objp[13] = (4.7625, 4.7625, 0)
    objp[14] = (4.7625, 7.9375, 0)
    objp[15] = (4.7625, 11.1125, 0)
    objp[16] = (6.35, 0  , 0)
    objp[17] = (6.35, 3.175 , 0)
    objp[18] = (6.35, 6.35, 0)
    objp[19] = (6.35, 9.525, 0)
    objp[20] = (7.9375, 1.5875 , 0)
    objp[21] = (7.9375, 4.7625, 0)
    objp[22] = (7.9375, 7.9375, 0)
    objp[23] = (7.9375, 11.1125, 0)
    objp[24] = (9.525, 0  , 0)
    objp[25] = (9.525, 3.175 , 0)
    objp[26] = (9.525, 6.35, 0)
    objp[27] = (9.525, 9.525, 0)
    objp[28] = (11.1125, 1.5875 , 0)
    objp[29] = (11.1125, 4.7625, 0)
    objp[30] = (11.1125, 7.9375, 0)
    objp[31] = (11.1125, 11.1125, 0)
    objp[32] = (12.7, 0  , 0)
    objp[33] = (12.7, 3.175 , 0)
    objp[34] = (12.7, 6.35, 0)
    objp[35] = (12.7, 9.525, 0)
    objp[36] = (14.2875, 1.5875 , 0)
    objp[37] = (14.2875, 4.7625, 0)
    objp[38] = (14.2875, 7.9375, 0)
    objp[39] = (14.2875, 11.1125, 0)
    objp[40] = (15.875, 0  , 0)
    objp[41] = (15.875, 3.175 , 0)
    objp[42] = (15.875, 6.35, 0)
    objp[43] = (15.875, 9.525, 0)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv2.VideoCapture(0)
    found = 0
    while(found < 10):
        ret,img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints = blobDetector.detect(gray)
        
        # Draw detected blobs as red circles. This helps cv2.findCirclesGrid()
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findCirclesGrid(im_with_keypoints, (4,11), None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(im_with_keypoints_gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)
            
            # draw and display the corners
            im_with_keypoints = cv2.drawChessboardCorners(img, (4,11), corners2, ret)
            found += 1
            
        cv2.imshow("img", im_with_keypoints) # display
        cv2.waitKey(2)
        
    # release the capture
    cap.release()
    cv2.destroyAllWindows()

    ret, matx, dist_coef, rvecs, t_vecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    t_vecs = np.mean(t_vecs, axis=0)
    global mtx, dist, tvecs
    mtx = matx
    dist = dist_coef
    tvecs = t_vecs
    return mtx, dist, tvecs
    
def calc_obj_distance(img_left, img_right, coord, camera_matrix, dist_coef, T):
    '''
    camera_matrix = np.array([[332.10175235, 0, 265.58673036], [0, 244.60821122, 273.70400038], [0, 0, 1]])
    dist_coef = np.array([[-2.08756630e+01,  2.44410090e+02,  8.28673330e-01, -9.08185175e-02, -8.22048678e+02]])
    T = np.array([-19.817797355554617, 3.07753175, 20.73977698])'''
    
    # rectification
    h,  w = img_left.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coef, (w,h), 1, (w,h))
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coef, None, newcameramtx, (w,h), 5)
    dst = cv2.remap(img_left, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst_left = dst[y:y+h, x:x+w]

    # rectification
    hr,  wr = img_right.shape[:2]
    newcameramtx_r, roi_r = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coef, (wr,hr), 1, (wr,hr))
    # undistort
    mapx_r, mapy_r = cv2.initUndistortRectifyMap(camera_matrix, dist_coef, None, newcameramtx_r, (wr,hr), 5)
    dst_r = cv2.remap(img_right, mapx_r, mapy_r, cv2.INTER_LINEAR)
    # crop the image
    xr, yr, wr, hr = roi_r
    dst_right = dst_r[yr:yr+hr, xr:xr+wr]
    
    # stereo correspondence
    window_size = 5
    min_disp = 16
    num_disp = 192-min_disp
    blockSize = window_size
    uniquenessRatio = 1
    speckleRange = 3
    speckleWindowSize = 3
    disp12MaxDiff = 200
    P1 = 600
    P2 = 2400
    stereo = cv2.StereoSGBM_create(
            minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = window_size,
            uniquenessRatio = uniquenessRatio,
            speckleRange = speckleRange,
            speckleWindowSize = speckleWindowSize,
            disp12MaxDiff = disp12MaxDiff,
            P1 = P1,
            P2 = P2
        )
    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    disparity = (disparity-min_disp)/num_disp # in pixels
    #print("Disparity ", disparity.shape)

    # distance calc
    focal_len = (camera_matrix[0,0] + camera_matrix[1][1])/2   # in pixels
    #print("focal length ", focal_len)
    baseline=abs(T[0]) # this is in the unit of one chess square
    #baseline *= 2 # approximately 2mm as the square width
    #print("baseline ", baseline)
    distance = ((focal_len * baseline)/ disparity[coord[1],coord[0]])/100 # this should be in m
    #print("distance: ", distance)
    return distance


@smart_inference_mode()
def run_yolo(
        weights= ROOT / 'yolov5/yolov5s.pt',  # model path or triton URL
        source=0,      #(webcam),
        data= ROOT / 'yolov5/data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    found = False

    webcam = True
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)    
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
        bs = len(dataset)

    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s, cap in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    s += f"{int(c)}"

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    if names[c] == 'person': # It's a person
                        found = True
                        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        annotator.box_label(xyxy, f'{names[c]}', color=colors(c,True))
                        x_center = int(p1[0] + ((p2[0]-p1[0]) / 2))
                        y_center = int(p1[1] + ((p1[1]+p2[1]) / 2))
                        #LOGGER.info(f'coords {x_center} {y_center}')
                        # Take two pictures few cm apart. For now, just pause
                        cv2.imwrite('/home/pi/distance_compute/img_left.jpg', im0)
                        #print('captured left')
                        # create a delay
                        wait=10
                        while(wait != 0):
                            wait = wait-1
                        # take the second picture
                        _, frame2 = cap.read()
                        if _:
                            cv2.imwrite('/home/pi/distance_compute/img_right.jpg', frame2)
                        img_left = cv2.imread('/home/pi/distance_compute/img_left.jpg')
                        img_right = cv2.imread('/home/pi/distance_compute/img_right.jpg')
                        distance = calc_obj_distance(img_left, img_right, [x_center, y_center], mtx, dist, tvecs)
                        LOGGER.info(f"Distance{colorstr('bold', distance)}")
                        annotator.box_label(xyxy, f'{names[c]} {distance}', color=colors(c,True))
                    #annotator.box_label(xyxy, f'{names[c]}', color=colors(c,True))

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    LOGGER.info(f'here')
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    return found, x_center, y_center, distance
