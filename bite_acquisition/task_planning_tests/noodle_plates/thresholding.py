
import torch
import cv2
import numpy as np
import os

# add parent directory to path
import sys
sys.path.append('../..')

from scripts.src.spaghetti_segmentation.model import SegModel
from scripts.vision_utils import resize_to_square, detect_plate, detect_blue, cleanup_mask, detect_densest, calculate_heatmap_density, calculate_heatmap_entropy
import torchvision.transforms as transforms

BASE_DIR = '/home/isacc/bite_acquisition'
PATH_TO_SPAGHETTI_CHECKPOINTS = BASE_DIR + '/spaghetti_checkpoints'
PARAM_TUNING = True # tune thresholding parameters using training data

if __name__ == '__main__':

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seg_net = SegModel("FPN", "resnet34", in_channels=3, out_classes=1)
    ckpt = torch.load('%s/spaghetti_seg_resnet.pth'%PATH_TO_SPAGHETTI_CHECKPOINTS, map_location=DEVICE)
    seg_net.load_state_dict(ckpt)
    seg_net.eval()
    seg_net.to(DEVICE)
    seg_net_transform = transforms.Compose([transforms.ToTensor()])

    if PARAM_TUNING:
        INPUT_DIR = BASE_DIR + '/task_planning_tests/noodle_plates/log/spaghetti/classification_format/train'
        OUTPUT_DIR = BASE_DIR + '/task_planning_tests/noodle_plates/param_tuning/thresholding/spaghetti'
        PREDICTION_DIR = OUTPUT_DIR + '/predictions.txt'
    else:
        INPUT_DIR = BASE_DIR + '/task_planning_tests/noodle_plates/log/spaghetti/classification_format/test'
        OUTPUT_DIR = BASE_DIR + '/task_planning_tests/noodle_plates/outputs/thresholding/spaghetti'
        PREDICTION_DIR = OUTPUT_DIR + '/predictions.txt'

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    CANDIDATE_ACTIONS = ['group', 'twirl']
    correct_count = 0
    total_valid_count = 0
    total_count = 0

    prediction_file = open(PREDICTION_DIR, 'w')
    for action_label in CANDIDATE_ACTIONS:
        if not os.path.exists(OUTPUT_DIR + '/' + action_label):
            os.makedirs(OUTPUT_DIR + '/' + action_label)
        TEST_IMGS = os.listdir(os.path.join(INPUT_DIR, action_label))
        # sort images by number
        TEST_IMGS.sort(key=lambda x: int(x.split('.')[0]))
        for img_name in TEST_IMGS:
            SOURCE_IMAGE_PATH = os.path.join(INPUT_DIR, action_label, img_name)
            print(SOURCE_IMAGE_PATH)
            image = cv2.imread(SOURCE_IMAGE_PATH)
            image = resize_to_square(image, 480)
            
            total_count += 1
            try:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                inp = seg_net_transform(img_rgb).to(device=DEVICE)
                logits = seg_net(inp)
                H,W,C = image.shape
                pr_mask = logits.sigmoid().detach().cpu().numpy().reshape(H,W,1)
                noodle_vapors_mask = cv2.normalize(pr_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
                clean_mask = noodle_vapors_mask
                
                densest, heatmap = detect_densest(clean_mask)
                heatmap = cv2.bitwise_and(heatmap, clean_mask)
                density = calculate_heatmap_density(heatmap)
                entropy = calculate_heatmap_entropy(heatmap)

                DENSITY_THRESH = 0.64
                ENTROPY_THRESH = 9.0

                total_valid_count += 1
                if density > DENSITY_THRESH or entropy <= ENTROPY_THRESH:
                    prediction_file.write(img_name + ' Label: ' + action_label + ' Prediction: twirl\n')
                    if action_label == 'twirl':
                        correct_count += 1
                else:
                    prediction_file.write(img_name + ' Label: ' + action_label + ' Prediction: group\n')
                    if action_label == 'group':
                        correct_count += 1
                    

                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)
                heatmap = cv2.putText(heatmap, 'Density Score: %.2f'%(density), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                heatmap = cv2.putText(heatmap, 'Entropy Score: %.2f'%(entropy), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

                cv2.imwrite(os.path.join(OUTPUT_DIR + '/' + action_label + '/', img_name), np.hstack([image, cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR), heatmap]))

            except Exception as e:
                print("Error:", e)
                continue
    
    prediction_file.close()
            
    print("Correct Count:", correct_count)
    print("Total Valid Count:", total_valid_count)
    print("Total Count:", total_count)


