# readme
# only when you are first, follow this step to set the environment.
################################
# to implement, go to INSTALL.md
# follow step 2 and 3 subsequently on pycharm terminal.
# change pycharm interpreter to lane-det activated in step 2.
################################
# On pycharm terminal, copy & paste this line.
# activate lane-det
# python LaneDetection.py configs/tusimple.py --test_model tusimple_18.pth
# use tusimple model.

import cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageGrab
from data.constant import tusimple_row_anchor

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('LaneDetection Started..')
    dist_print('')
    dist_print("press 'q' to stop.")
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_w, img_h = 1280, 720
    row_anchor = tusimple_row_anchor
    cnt = 0
    while True:
        cnt += 1
        if cnt == 200:
            break
        original_img = ImageGrab.grab(bbox=(0, 40, img_w, img_h))
        # we have to first transform original_img
        # transforms img to Tensor; torchSize: [3, 288, 800], type: torch.Tensor
        imgs = img_transforms(original_img)
        # add first dimension for batch size. torchSize: [1, 3, 288, 800]
        imgs = torch.unsqueeze(imgs, 0)
        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc
        original_img = np.array(original_img)
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        # add circles representing two lanes ot original image.
                        cv2.circle(original_img,ppp,5,(0,255,0),-1)

        cv2.imshow("Frame", original_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
