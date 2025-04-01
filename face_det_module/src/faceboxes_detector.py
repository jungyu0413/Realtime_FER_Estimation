import cv2
import torch
from face_det_module.src.utils.dedector import Detector
from face_det_module.src.utils.config import cfg
from face_det_module.src.utils.prior_box import PriorBox
from face_det_module.src.utils.faceboxes import FaceBoxesV2
from face_det_module.src.utils.box_utils import decode
from PIL import Image

class FaceBoxesDetector(Detector):
    def __init__(self, model_arch, model_weights, use_gpu, device):
        super().__init__(model_arch, model_weights)
        self.name = 'FaceBoxesDetector'
        self.net = FaceBoxesV2(phase='test', size=None, num_classes=2)
        self.use_gpu = use_gpu
        self.device = device

        state_dict = torch.load(self.model_weights, map_location=self.device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        self.net = self.net.to(self.device)
        self.net.eval()

    def detect(self, image, thresh_xy=600, thresh=0.8, type='max', im_scale=None):
        if im_scale is None:
            height, width, _ = image.shape
            if min(height, width) > thresh_xy:
                im_scale = thresh_xy / min(height, width)
            else:
                im_scale = 1

        image_scale = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        scale = torch.Tensor([image_scale.shape[1], image_scale.shape[0], image_scale.shape[1], image_scale.shape[0]])
        image_scale = torch.from_numpy(image_scale.transpose(2, 0, 1)).to(self.device).int()
        mean_tmp = torch.IntTensor([104, 117, 123]).to(self.device).unsqueeze(1).unsqueeze(2)
        image_scale -= mean_tmp
        image_scale = image_scale.float().unsqueeze(0)
        scale = scale.to(self.device)

        with torch.no_grad():
            out = self.net(image_scale)
            priorbox = PriorBox(cfg, image_size=(image_scale.size()[2], image_scale.size()[3]))
            priors = priorbox.forward().to(self.device)
            loc, conf = out
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance']) * scale
            scores = conf.data[:, 1]

            box_dis = torch.tensor([i[2] - i[0] for i in boxes])
            bbox_thresh = torch.tensor(100)
            inds_bbox = torch.where(box_dis >= bbox_thresh)[0]
            boxes = boxes[inds_bbox]
            scores = scores[inds_bbox]

            if type == 'max':
                th_max = torch.max(scores)
                if th_max < thresh:
                    return False, False
                else:
                    inds = torch.where(scores >= thresh)[0]
                    boxes = boxes[inds]
                    scores = scores[inds]

                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    max_idx = torch.argmax(areas)
                    dets = boxes[max_idx].unsqueeze(0)
                    score = scores[max_idx].item()

                    detections_scale = []
                    xmin = int(dets[0][0])
                    ymin = int(dets[0][1])
                    xmax = int(dets[0][2])
                    ymax = int(dets[0][3])
                    width = xmax - xmin
                    height = ymax - ymin
                    detections_scale.append(['face', score, xmin, ymin, width, height])
                    check = True

                    if len(detections_scale) > 0:
                        detections_scale = [[det[0], det[1], int(det[2] / im_scale), int(det[3] / im_scale), int(det[4] / im_scale), int(det[5] / im_scale)] for det in detections_scale]

        return detections_scale, check