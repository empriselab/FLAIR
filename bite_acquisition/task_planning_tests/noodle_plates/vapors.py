
import os
import sys

BASE_DIR = '/home/isacc/bite_acquisition'

if __name__ == '__main__':
    
    INPUT_DIR = BASE_DIR + '/task_planning_tests/noodle_plates/log/spaghetti/classification_format/test'
    OUTPUT_DIR = BASE_DIR + '/task_planning_tests/noodle_plates/outputs/vapors/spaghetti'
    PREDICTION_DIR = OUTPUT_DIR + '/predictions.txt'

    ANNOTATION_FILE = BASE_DIR + '/task_planning_tests/noodle_plates/vapors_labels.txt'

    # read from annotation file
    vapors_annotations = []
    with open(ANNOTATION_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = line[:-1]
            vapors_annotations.append(label)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    CANDIDATE_ACTIONS = ['group', 'twirl']
    correct_count = 0
    total_count = 0

    prediction_file = open(PREDICTION_DIR, 'w')
    for action_label in CANDIDATE_ACTIONS:
        if not os.path.exists(OUTPUT_DIR + '/' + action_label):
            os.makedirs(OUTPUT_DIR + '/' + action_label)
        TEST_IMGS = os.listdir(os.path.join(INPUT_DIR, action_label))
        # sort images by number
        TEST_IMGS.sort(key=lambda x: int(x.split('.')[0]))
        for img_name in TEST_IMGS:
            img_num = img_name.split('.')[0]
            vapors_annotation = vapors_annotations[int(img_num)-1]
            prediction_file.write(img_name + ' Label: ' + action_label + ' Prediction: ' + vapors_annotation + '\n')
            if action_label == vapors_annotation:
                correct_count += 1
            total_count += 1
    
    prediction_file.close()
            
    print("Correct Count:", correct_count)
    print("Total Count:", total_count)


