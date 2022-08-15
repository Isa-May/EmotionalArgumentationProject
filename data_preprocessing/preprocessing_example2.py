import argparse
import os

import pandas
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATASET_FOLDER = os.path.join('../../data/master_files')
PROCESSED_DATASET_FOLDER = os.path.join('../../data/master_files/out')


def generate_dataset_with_labels(raw_dataset_folder, output_folder):
    label_cols = ['id', 'sent', 'source', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    entire_label_set = pandas.read_csv(raw_dataset_folder + os.path.join('/semev07_master.csv'),
                                       sep=',', header=0, usecols=label_cols, skipinitialspace=True)

    keys = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    label_list = entire_label_set.values.tolist()
    print(label_list)
    # make list out of columns with emotions
    n = 3
    # delete first three items in each tupel
    new_list = []
    for tupel in label_list:
        new_element = tupel[n:]
        new_list.append(new_element)
    print(new_list)

    # create dict from new_list and keys

    new_list_dict = []

    for tupel in new_list:
        new_element_2 = zip(keys, tupel)
        new_element_2 = dict(new_element_2)
        new_list_dict.append(new_element_2)
    print(new_list_dict)

    max_list = []
    for dic_item in new_list_dict:
        maximum = max(dic_item.values())
        newDict = {key: value for (key, value) in dic_item.items() if value == maximum}
        max_list.append(newDict)
    # we now have a list of dict values containing the emotion with the highest value

    # from here create a new list consisting only of the emotions
    emotion_list = []
    for element in max_list:
        emotion_list_el = list(element.keys())
        emotion_list.append(emotion_list_el)

    # unpack the list elements which are currently lists so they are only values
    emotion_list_bare = []
    for element in emotion_list:
        emotion_list_bare_el = element[0]
        emotion_list_bare.append(emotion_list_bare_el)

    print(emotion_list_bare)
    # go through list of emotions and replace them by their respective encoding, then transform to dataframe called 'emotion'
    emo_list_final = [y for y in [
        x.replace('anger', '1').replace('disgust', '2').replace('fear', '3').replace('joy', '6').replace('sadness',
                                                                                                         '4').replace(
            'surprise', '0') for x in emotion_list_bare]]
    results = [int(i) for i in emo_list_final]
    print(results)

    # convert results list to dataframe which is the emotion column
    emo_df = pd.DataFrame(results, columns=['Emotion'])
    entire_label_set['Emotion'] = emo_df

    print(entire_label_set['Emotion'])

    entire_label_set.pop('id')
    entire_label_set.pop('anger')
    entire_label_set.pop('disgust')
    entire_label_set.pop('fear')
    entire_label_set.pop('joy')
    entire_label_set.pop('sadness')
    entire_label_set.pop('surprise')

    # rename headlines
    mapping = {entire_label_set.columns[0]: 'Sentence'}
    entire_label_set = entire_label_set.rename(columns=mapping)

    mapping = {entire_label_set.columns[1]: 'Source'}
    entire_label_set = entire_label_set.rename(columns=mapping)

    mapping = {entire_label_set.columns[2]: 'Emotion'}
    entire_label_set = entire_label_set.rename(columns=mapping)

    entire_label_set = entire_label_set.drop_duplicates(subset=['Sentence'])

    entire_label_set = entire_label_set.reset_index(drop=True)
    entire_label_set.index.name = 'Id'

    # create train, dev, test splits
    # first create 80%-sized train and 20 % test_entire, from the latter make 10% dev and test in a 2nd step
    semeval_2007 = entire_label_set
    entire_label_set.to_csv(output_folder + os.path.join("/semev07_master_new.csv"))

    y = semeval_2007['Emotion']
    X = semeval_2007

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train['Set'] = 'train'

    X_test['Set'] = 'test'

    # from test set generate 50 per cent dev and 50 per cent train
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
    X_train_dev.to_csv(output_folder + os.path.join("/semev07_master_train_dev.csv"))

    X_train.to_csv(output_folder + os.path.join("/semev07_master_train.csv"))
    X_dev.to_csv(output_folder + os.path.join("/semev07_master_dev.csv"))
    X_test_final.to_csv(output_folder + os.path.join("/semev07_master_test.csv"))


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
