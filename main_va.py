import cv2
import numpy as np
import torch
from torchvision import transforms
from VA_module.model import Model
from face_det_module.src.util import get_args_parser, get_transform, pre_trained_wegiths_load
from face_det_module.src.face_crop import crop
from face_det_module.src.util import resize_image

font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
v_font_color = (0, 255, 0)  
a_font_color = (0, 0, 255)  
font_thickness = 2
resize_param = 250

if __name__ == "__main__":
    args = get_args_parser()
    args.transform = get_transform()
    args.weights_path = '/NLA/Valence_Arousal/VA_module/weights/best.pth'
    model = Model(args)
    cp = torch.load(args.weights_path)
    weighted_Model = pre_trained_wegiths_load(model, cp)
    model = model.to('cuda')
    model.eval()
    VA = cv2.imread('/NLA/VA_module/VA.jpg')
    VA,  (new_center_h, new_center_w, new_length) = resize_image(VA, resize_param)
    height, width = VA.shape[:2]

    radius = 2
    color = (0, 0, 255)  # red
    thickness = -1
    idx = True
    cap = cv2.VideoCapture(0)
    preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 변경
    transforms.ToTensor(),  # PIL 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.577, 0.4494, 0.4001],
                         std=[0.2628, 0.2395, 0.2383])])
    while idx:
        idx, image = cap.read()
        image = cv2.flip(image, 1)
        img_height, img_width = image.shape[:2]
        output_image, check = crop(image, preprocess, 224, True, 'cuda')
        if check:
            _, pred_val, pred_aro, _, _ = model(output_image)
            
            valence =  np.clip(np.round((pred_val.item()), 2), -1, 1)
            arousal = np.clip(np.round((pred_aro.item()), 2), -1, 1) 
            map_val = int(valence * (new_length/2))
            map_aro = int(arousal * (new_length/2))
            
            cv2.putText(image, "Valence: " + str(valence), (10, 50), font, font_size, v_font_color, font_thickness)
            cv2.putText(image, "Arousal: " + str(arousal), (10, 100), font, font_size, a_font_color, font_thickness)
        else:
            cv2.putText(image, "Valence: " + 'None', (10, 50), font, font_size, v_font_color, font_thickness)
            cv2.putText(image, "Arousal: " + 'None', (10, 100), font, font_size, a_font_color, font_thickness)
            
        image[img_height-height:, img_width-width:] = VA
        
        
        cent_h, cent_w = int(img_height-(height/2)), int(img_width-(width/2))
        cv2.circle(image, (cent_w+map_val, cent_h-map_aro), radius, color, thickness)
        #print(map_val, map_aro)
        


        
        if idx == False:
            cap.release()
        else:
            cv2.imshow("Output", image)
            k = cv2.waitKey(2) & 0xFF
            if k == 27: # ESC key
                break