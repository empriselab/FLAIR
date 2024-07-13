import os

OUTPUT_DIR = '/home/isacc/bite_acquisition/task_planning_tests/noodle_plates/outputs'
# METHODS = ['thresholding', 'gpt', 'swin_transformer', 'vapors']
METHODS = ['thresholding', 'gpt', 'vapors']

for method in METHODS:
    prediction_file = f'{OUTPUT_DIR}/{method}/spaghetti/predictions.txt'
    total_count = 0
    correct_count = 0
    with open(prediction_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            image_name = line.split(' ')[0]
            label = line.split(' ')[2]
            prediction = line.split(' ')[4][:-1] # remove newline character
            if label == prediction:
                correct_count += 1
            total_count += 1

    print(f'Method: {method} Total: {total_count} Correct: {correct_count} Accuracy: {correct_count/total_count}')