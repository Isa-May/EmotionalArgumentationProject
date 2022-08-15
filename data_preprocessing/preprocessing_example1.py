import argparse
import os

import numpy as np
import pandas
from sklearn.model_selection import train_test_split

RAW_DATASET_FOLDER = os.path.join('../../data/master_files')
PROCESSED_DATASET_FOLDER = os.path.join('../../data/master_files/out')


def generate_dataset_with_labels(raw_dataset_folder, output_folder):
    label_cols = ['id', 'sent', 'source', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'SurprisedPl',
                  'SurprisedMin']
    entire_label_set = pandas.read_csv(raw_dataset_folder + os.path.join('/alm_master.csv'),
                                       sep=',', header=0, usecols=label_cols, skipinitialspace=True)

    entire_label_set = entire_label_set[entire_label_set['SurprisedPl'] == 0]
    entire_label_set = entire_label_set[entire_label_set['SurprisedMin'] == 0]

    entire_label_set.pop('SurprisedPl')
    entire_label_set.pop('SurprisedMin')

    happiness_column = np.where(entire_label_set["happiness"] == 1, 5, 0)
    entire_label_set['happiness'] = happiness_column

    angry_column = np.where(entire_label_set["anger"] == 1, 1, 0)
    entire_label_set['anger'] = angry_column

    disgust_column = np.where(entire_label_set["disgust"] == 1, 2, 0)
    entire_label_set['disgust'] = disgust_column

    fear_column = np.where(entire_label_set["fear"] == 1, 3, 0)
    entire_label_set['fear'] = fear_column

    sadness_column = np.where(entire_label_set["sadness"] == 1, 4, 0)
    entire_label_set['sadness'] = sadness_column

    test = zip(entire_label_set.happiness, entire_label_set.anger, entire_label_set.disgust, entire_label_set.fear,
               entire_label_set.sadness)
    test = list(test)

    # iterate through list, if only 0 return 0 if single element different return different element
    # check if all elements are equal
    my_list = []
    for entry in test:
        result = all(element == 0 for element in entry)
        if result:
            new_element = 0
            my_list.append(new_element)
        else:
            # for item in entry:
            new_element_2 = [x for x in entry if x != 0]
            # get out of list
            new_element_2 = new_element_2[0]
            my_list.append(new_element_2)

    print(my_list)

    entire_label_set['Emotion'] = my_list

    print(entire_label_set['Emotion'])

    entire_label_set.pop('id')
    entire_label_set.pop('anger')
    entire_label_set.pop('disgust')
    entire_label_set.pop('fear')
    entire_label_set.pop('happiness')
    entire_label_set.pop('sadness')

    # rename headlines
    mapping = {entire_label_set.columns[0]: 'Sentence'}
    entire_label_set = entire_label_set.rename(columns=mapping)

    mapping = {entire_label_set.columns[1]: 'Source'}
    entire_label_set = entire_label_set.rename(columns=mapping)

    mapping = {entire_label_set.columns[2]: 'Emotion'}
    entire_label_set = entire_label_set.rename(columns=mapping)

    dictionary = {'"': '', '\*': ''}
    entire_label_set.replace(dictionary, regex=True, inplace=True)
    entire_label_set = entire_label_set.drop_duplicates(subset=['Sentence'])

    entire_label_set = entire_label_set.reset_index(drop=True)
    entire_label_set.index.name = 'Id'

    alm = entire_label_set

    y = alm['Emotion']
    X = alm

    # use stratified sampling to create train/dev/test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train['Set'] = 'train'

    X_test['Set'] = 'test'

    test_set_all = X_test

    y = test_set_all['Emotion']
    X = test_set_all

    X_test_final, X_dev, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    X_test_final['Set'] = 'test'

    X_dev['Set'] = 'dev'

    X_train = X_train[['Sentence', 'Source', 'Set', 'Emotion']]
    X_train = X_train.reset_index(drop=True)
    X_train.index.name = 'Id'

    X_dev = X_dev[['Sentence', 'Source', 'Set', 'Emotion']]
    X_dev = X_dev.reset_index(drop=True)
    X_dev.index.name = 'Id'

    X_test_final = X_test_final[['Sentence', 'Source', 'Set', 'Emotion']]
    X_test_final = X_test_final.reset_index(drop=True)
    X_test_final.index.name = 'Id'

    X_train_dev = [X_train, X_dev]
    X_train_dev = pandas.concat(X_train_dev)
    X_train_dev.to_csv(output_folder + os.path.join("/alm_master_train_dev.csv"))

    X_train.to_csv(output_folder + os.path.join("/alm_master_train.csv"))
    X_dev.to_csv(output_folder + os.path.join("/alm_master_dev.csv"))
    X_test_final.to_csv(output_folder + os.path.join("/alm_master_test.csv"))


def generate_csv_dataset():
    parser = argparse.ArgumentParser(description='Preprocess the master emotions files.')
    parser.add_argument('--raw', type=str, help='the path where the raw emotion dataset is located.',
                        default=RAW_DATASET_FOLDER)
    parser.add_argument('--output', type=str, help='the path where the generated csv should be stored.)',
                        default=PROCESSED_DATASET_FOLDER)

    args = parser.parse_args()
    generate_dataset_with_labels(args.raw, args.output)


if __name__ == '__main__':
    generate_csv_dataset()
