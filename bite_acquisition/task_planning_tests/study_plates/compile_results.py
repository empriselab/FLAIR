import os

OUTPUT_DIR = '/home/isacc/bite_acquisition/task_planning_tests/study_plates/outputs'
# METHODS = ['thresholding', 'gpt', 'swin_transformer', 'vapors']
METHODS = ['thresholding', 'gpt']

PLATES = ['spaghetti_meatballs', 'fettuccine_chicken_broccoli', 'mashed_potato_sausage', 'oatmeal_strawberry', 'dessert']

for method in METHODS:
    accuracies = []
    for plate in PLATES:
        prediction_file = f'{OUTPUT_DIR}/{method}/{plate}/predictions.txt'
        total_count = 0
        correct_count = 0
        with open(prediction_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_name = line.split(' ')[0]
                label = line.split(' ')[2]
                if len(line.split(' ')) == 5:
                    prediction = line.split(' ')[4][:-1] # remove newline character
                else:
                    prediction = line.split(' ')[4]
                if (prediction == 'Acquire' and label == 'twirl') or (prediction == 'Group' and label == 'group') or (prediction == 'Push' and label == 'push'):
                    correct_count += 1
                elif (prediction == 'Acquire' and label == 'scoop') or (prediction == 'Push' and label == 'push'):
                    correct_count += 1
                elif label == prediction:
                    correct_count += 1
                
                total_count += 1
        print(f'Method: {method} Total: {total_count} Correct: {correct_count} Accuracy: {correct_count/total_count}')
        accuracies.append(correct_count/total_count)
    print(f'Method: {method} Average Accuracy: {sum(accuracies)/len(accuracies)}')
