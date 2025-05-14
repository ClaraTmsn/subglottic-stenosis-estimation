import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from motrackers import IOUTracker
from motrackers.utils import draw_tracks
import os
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator


threshold_area = 10000

broncos = [15] # Video id
starts =  [190] # Frame to start tracking, when endoscope is about to enter vocal cords


for b in range(len(broncos)):
    bboxes = []
    masks = []
    confidences = []
    class_ids = []
    foundFirst = False
    foundFrame = False
    tracker = IOUTracker(max_lost=25, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
                             tracker_output_format='visdrone_challenge')

    frames_path = 'data/SGS_updated/bronco'+str(broncos[b])+'/frames/'
    out_path = 'data/SGS_updated/bronco'+str(broncos[b])+'/out_seg.mp4'
    images = sorted(os.listdir(frames_path))
    st = starts[b]
    out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (1120,800))

    ## Before start tracking
   
    if st>0:
        for img_name in images[:st]:

            img = cv2.imread(frames_path+img_name)

            #img = img[:,10:img.shape[1]-20,:] ## crop 1: endoscopes 1&3
            #img = img[20:img.shape[0]-20,190:img.shape[1]-210,:] ## crop 1: endoscope 2

            width, height = img.shape[1], img.shape[0]
            crop_width, crop_height = (1120,800)
            mid_x, mid_y = int(width/2), int(height/2)
            cw2, ch2 = int(crop_width/2), int(crop_height/2) 
            img = img[mid_y-ch2:mid_y+ch2, max(0,mid_x-cw2):mid_x+cw2]
            img = cv2.resize(img, (crop_width,crop_height))

            out.write(img)
    
    
    img_name = images[st]

    ## First tracking frame = "Pedal pressed"

    img = cv2.imread(frames_path+img_name)

    #img = img[:,10:img.shape[1]-20,:] ## crop 1: endoscopes 1&3
    #img = img[20:img.shape[0]-20,190:img.shape[1]-210,:] ## crop 1: endoscope 2

    width, height = img.shape[1], img.shape[0]
    crop_width, crop_height = (1120,800)
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    img = img[mid_y-ch2:mid_y+ch2, max(0,mid_x-cw2):mid_x+cw2]
    img = cv2.resize(img, (crop_width,crop_height))

    img_copy = img.copy()

    mask_ref = img.copy()
    mask_ref = cv2.cvtColor(mask_ref, cv2.COLOR_BGR2GRAY)
    mask_ref = cv2.normalize(mask_ref, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, thresholded = cv2.threshold(mask_ref, 50, 255, cv2.THRESH_BINARY)
    thresholded = 255 - thresholded

    (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(thresholded)

    index = 1
    fistFoldProc = False
    min_i = np.unravel_index(np.argmin(mask_ref), mask_ref.shape)
    masks = []
    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]

        if area > threshold_area:       
                 
            x = values[i, cv2.CC_STAT_LEFT]
            y = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]
            mask = thresholded[y:y+h,x:x+w]
            mask = cv2.resize(mask, (1120,800))
            if min_i[1]>=x and min_i[1]<= (x+w) and min_i[0]>=y and min_i[0]<= (y+h): 
                if not fistFoldProc:
                        bboxes = np.array([[x, y, w, h]], np.int32)
                        confidences = np.array([1.0], np.float32)
                        class_ids = np.array([1], np.int32)
                        fistFoldProc = True
                        masks = [mask]
                else:
                        bboxes = np.append(bboxes, [[x, y, w, h]], axis=0)
                        confidences = np.append(confidences, [1.0], axis=0)
                        class_ids = np.append(class_ids, [1], axis=0)
                        masks.append(mask)
                
        index = index + 1

    # Start tracking

    tracks = tracker.update(bboxes, confidences, class_ids)

    # Display the output contours and bounding rectangles
    for i in range(1, totalLabels):
        marked = False
        for indexTrack, itemTrack in enumerate(tracks):
            x = values[i, cv2.CC_STAT_LEFT]
            y = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]

            if itemTrack[2] == x and  itemTrack[3] == y and itemTrack[4] == w and itemTrack[5] == h:
                img[label_ids == i] = (0,255,0) 
                marked = True
                cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2) 
                cv2.circle(img, (int(centroid[i][0]),int(centroid[i][1])), 2, (0,0,0), -1)
                cv2.putText(img, 'C', (int(centroid[i][0]),int(centroid[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        if not marked:
            img[label_ids == i] = (255,0,0) 
         
    img = draw_tracks(img, tracks)
    cv2.circle(img, (min_i[1],min_i[0]), 10, (0,0,255), -1)      
    alpha = 0.4  # Transparency factor.
    # Following line overlays transparent rectangle over the image
    image_alpha = cv2.addWeighted(img, alpha,img_copy , 1 - alpha, 0)

    cv2.rectangle(image_alpha, (10,10), (image_alpha.shape[1]-10, image_alpha.shape[0]-10), (0,0,255), 2) 
    cv2.putText(image_alpha, 'Pedal pressed, tracking activated', (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
    for t in range(25):
        out.write(image_alpha)
    
    ## Next frames
    for img_name in images[st+1:]:
        img = cv2.imread(frames_path+img_name)
        
        #img = img[:,10:img.shape[1]-20,:] ## crop 1: endoscopes 1&3
        #img = img[20:img.shape[0]-20,190:img.shape[1]-210,:] ## crop 1: endoscope 2

        width, height = img.shape[1], img.shape[0]
        crop_width, crop_height = (1120,800)
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        img = img[mid_y-ch2:mid_y+ch2, max(0,mid_x-cw2):mid_x+cw2]
        
        img = cv2.resize(img, (crop_width,crop_height))
        cv2.imwrite('bronco'+str(broncos[b])+'/frames_cut/'+img_name,img)
        img_copy = img.copy()

        mask_ref = img.copy()
        mask_ref = cv2.cvtColor(mask_ref, cv2.COLOR_BGR2GRAY)
        mask_ref = cv2.normalize(mask_ref, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, thresholded = cv2.threshold(mask_ref, 50, 255, cv2.THRESH_BINARY)
        thresholded = 255 - thresholded

        (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(thresholded)

        index = 1
        fistFoldProc = False
        min_i = np.unravel_index(np.argmin(mask_ref), mask_ref.shape)

        for i in range(1, totalLabels):
            area = values[i, cv2.CC_STAT_AREA]
            if area > threshold_area:       
                    
                x = values[i, cv2.CC_STAT_LEFT]
                y = values[i, cv2.CC_STAT_TOP]
                w = values[i, cv2.CC_STAT_WIDTH]
                h = values[i, cv2.CC_STAT_HEIGHT]
                mask = thresholded[y:y+h,x:x+w]
                mask = cv2.resize(mask, (1120,800))
                if min_i[1]>=x and min_i[1]<= (x+w) and min_i[0]>=y and min_i[0]<= (y+h):
                    if not fistFoldProc:
                            bboxes = np.array([[x, y, w, h]], np.int32)
                            confidences = np.array([1.0], np.float32)
                            class_ids = np.array([1], np.int32)
                            fistFoldProc = True
                            masks = [mask]
                    else:
                            bboxes = np.append(bboxes, [[x, y, w, h]], axis=0)
                            confidences = np.append(confidences, [1.0], axis=0)
                            class_ids = np.append(class_ids, [1], axis=0)
                            masks.append(mask)
                    
            index = index + 1
        # Start tracking

        tracks = tracker.update(bboxes, confidences, class_ids)

        # Display the output contours and bounding rectangles
        for i in range(1, totalLabels):
            marked = False
            for indexTrack, itemTrack in enumerate(tracks):
                x = values[i, cv2.CC_STAT_LEFT]
                y = values[i, cv2.CC_STAT_TOP]
                w = values[i, cv2.CC_STAT_WIDTH]
                h = values[i, cv2.CC_STAT_HEIGHT]

                if itemTrack[2] == x and  itemTrack[3] == y and itemTrack[4] == w and itemTrack[5] == h:
                    img[label_ids == i] = (0,255,0) 
                    marked = True
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2) 
                    cv2.circle(img, (int(centroid[i][0]),int(centroid[i][1])), 2, (0,0,0), -1)
                    cv2.putText(img, 'C', (int(centroid[i][0]),int(centroid[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            if not marked:
                img[label_ids == i] = (255,0,0)            
        img = draw_tracks(img, tracks)
        alpha = 0.4  # Transparency factor.
        # Following line overlays transparent rectangle over the image
        if len(tracks)>0 and foundFirst==False:
            foundFirst = True
            print('First frame')
            image_alpha = cv2.addWeighted(img, alpha,img_copy , 1 - alpha, 0)
            cv2.rectangle(image_alpha, (10,10), (image_alpha.shape[1]-10, image_alpha.shape[0]-10), (0,0,255), 2) 
            cv2.putText(image_alpha, 'Pedal pressed, tracking activated', (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
            for t in range(25):
                out.write(image_alpha)
        elif len(tracks)==0 and foundFirst == True and foundFrame==False:  
            ind = int(img_name.replace('.png',''))
            print('Found frame: ' + str(ind))
            foundFrame = True
            cv2.rectangle(img, (10,10), (img.shape[1]-10, img.shape[0]-10), (0,0,255), 2) 
            cv2.putText(img, 'Frame found: '+str(ind), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
            image_alpha = cv2.addWeighted(img, alpha,img_copy , 1 - alpha, 0)
            for t in range(25):
                out.write(image_alpha)

        else:
            cv2.circle(img, (min_i[1],min_i[0]), 10, (0,0,255), -1) 
            image_alpha = cv2.addWeighted(img, alpha,img_copy , 1 - alpha, 0)
            out.write(image_alpha)
        
        cv2.imwrite('bronco'+str(broncos[b])+'/frames_seg/'+img_name,image_alpha)
        def mask_overlay(image, mask, color=(0, 255, 0)):
            """
            Helper function to visualize mask on the top of the car
            """
            mask = np.dstack((mask, mask, mask)) * np.array(color)
            mask = mask.astype(np.uint8)
            #print(mask)
            weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
            img = image.copy()
            ind = mask[:, :, :] > 0    
            img[ind] = weighted_sum[ind]    
            return img


