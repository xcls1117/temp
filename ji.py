import numpy as np
import cv2
import torch
import sys
sys.path.append('/project/train/src_repo')


from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox


label_id_map = {0: "large_luggage"}
weights = '/project/train/models/final/best.weights'
device_tmp = '0'
img_size = 960 
conf_thres = 0.55
iou_thres =0.45


def init():
    device = select_device(device_tmp)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    if half:
        model.half()
    return model


def detect(model, input_image):
    results = {'code':0, 'json': {"objects": []}}

    set_logging()
    device = select_device(device_tmp)

    # Load model
    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    half = device.type != 'cpu'

    # dataset = LoadImages(source, img_size=imgsz)
    img = letterbox(input_image, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference

    img_tmp = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img_tmp.half() if half else img_tmp) if device.type != 'cpu' else None  # run once
    # for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    ##############################################
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0 = input_image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            det = det.cpu().numpy()
            # Write results
            for x1, y1, x2, y2, conf, label in det:
                # print(x1, y1, x2, y2, conf, label)
                res = {"xmin": x1,
                       "ymin": y1,
                       "xmax": x2,
                       "ymax": y2,
                       "confidence": conf,
                       "name": label_id_map[int(label)]}

                results["json"]["objects"].append(res)
                # cv2.rectangle(input_image, (x1, y1), (x2, y2), thickness=2, color=(0,0,255))
                # cv2.imshow('', input_image)
                # cv2.waitKey()
        print(results)
        # json.dumps(results, indent=4)
        return results


if __name__ == '__main__':
    source = '/home/data/81/LSCGM_luggageAndBabyCar_79.jpg'
    input_image = cv2.imread(source)
    model = init()
    detect(model, input_image)

