import numpy as np
import cv2
import torch
import torch.nn as nn
import os
# from network import Model
# from dataloader import get_data_loader
import torch.cuda as cuda
from PIL import Image
import torchvision.transforms as transforms
import gc
import glob
from norfair import Detection, Tracker, Video, draw_points

image_folder = 'input/'
mask_folder = 'sv/'

# Norfair
# video = Video(input_path="video.mp4")
tracker = Tracker(distance_function="euclidean", distance_threshold=50)

# sort the files in the folder according to the number in the file name
files = sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0]))

for filename in files:
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_folder, filename)
        
        # Extract the file name without extension
        file_name = os.path.splitext(filename)[0]
        
        # Construct the output path with the desired extension
        mask_path = os.path.join(mask_folder, f'{file_name}.0.bmp')

        # if mask path doesnt exist go to next image
        if not os.path.exists(mask_path):
            continue
        else:
            target_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            # orig_image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
            orig_image1 = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)

        # find the different contors in the target image
        contours, _ = cv2.findContours(target_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get bounding box for each contour
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            else:
                x, y, w, h = cv2.boundingRect(contour)
                # store the centre values in a list
                detections.append([x+h/2, y+w/2])
        if detections == []:
            continue
        else:
            detections = np.array(detections)
            # detections = [Detection(detections)]
            detections = [Detection(p) for p in detections]
            tracked_objects = tracker.update(detections=detections)
            draw_points(orig_image1, tracked_objects)
            cv2.imwrite('output/'+filename, orig_image1)
        
    # crop the bounding box from the image
            # crop = orig_image[y:y + h, x:x + w]
            # crop1 = orig_image1[y:y + h, x:x + w]
    
            # model = Model()
            # model.load_state_dict(torch.load('weights'))
            # model.eval()
            # img =torch.from_numpy(crop1).permute(2,0,1).float()
            # if(1.0==model(img.unsqueeze(0))[0].item()):
            #     orig_image1 = cv2.rectangle(orig_image1, [x,y], (x+w,y+h), (0, 255, 0), 2)
            #     orig_image1 = cv2.putText(orig_image1,'herring',[x,y],cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),1, cv2.LINE_AA) 
            # else:
            #     orig_image1 = cv2.rectangle(orig_image1, [x,y], (x+w,y+h), (255,0, 0), 2)
            #     orig_image1 = cv2.putText(orig_image1,'non-herring',[x,y],cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1, cv2.LINE_AA) 
            # orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            # output_path = image_path = os.path.join('output/', filename)
            # cv2.imwrite(output_path, orig_image1)


#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     print(f'Number of objects detected: {object_count}')
# result = cv2.matchTemplate(target_image, template_image, cv2.TM_CCOEFF_NORMED)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# threshold = 0.7629
# locations = np.where(result >= threshold)

# object_count = 0

# for pt in zip(*locations[::-1]):
#     cv2.rectangle(target_image, pt, (pt[0] + template_image.shape[1], pt[1] + template_image.shape[0]), (0, 255, 0), 2)
#     object_count += 1
# target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(8, 6))
# plt.imshow(target_image_rgb)
# plt.title('Object Counting')
# plt.axis('off')
# plt.show()
# Display the target image with bounding boxes around detected objects
# cv2.imshow('Object Counting', target_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(f'Number of objects detected: {object_count}')
