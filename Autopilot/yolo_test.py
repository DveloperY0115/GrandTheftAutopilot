import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().eval()  # yolov5s.pt
model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS

# Inference
with torch.no_grad():
    prediction = model(imgs, size=640)  # includes NMS

# Plot
for i, (img, pred) in enumerate(zip(imgs, prediction)):
    str = 'Image %g/%g: %gx%g ' % (i + 1, len(imgs), *img.shape[:2])
    img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
    if pred is not None:
        for c in pred[:, -1].unique():
            n = (pred[:, -1] == c).sum()  # detections per class
            str += '%g %ss, ' % (n, model.names[int(c)])  # add to string
        for *box, conf, cls in pred:  # xyxy, confidence, class
            label = model.names[int(cls)] if hasattr(model, 'names') else 'class_%g' % cls
            # str += '%s %.2f, ' % (label, conf)  # label
            ImageDraw.Draw(img).rectangle(box, width=3)  # plot
    img.save('results%g.jpg' % i)  # save
    print(str + 'Done.')