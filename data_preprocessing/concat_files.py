import argparse
import os

import pandas

RAW_DATASET_FOLDER = '../../data_preprocessing/master_files/out/'
OUTPUT_DATASET_FOLDER = '../../data_preprocessing/master_files/multi'


def generate_entire_labelset(folder):
    semev07_tr_de = pandas.read_csv(folder.raw + 'semev07_master_train_dev.csv')
    semev18_tr_de = pandas.read_csv(folder.raw + 'semev18_master_train_dev.csv')
    semev19_tr_de = pandas.read_csv(folder.raw + 'semev19_master_train_dev.csv')
    alm_tr_de = pandas.read_csv(folder.raw + 'alm_master_train_dev.csv')
    isear_tr_de = pandas.read_csv(folder.raw + 'isear_master_train_dev.csv')
    annot_tr_de = pandas.read_csv(folder.raw + 'annot_master_train_dev.csv')

    combined_master_multi_tr_de = [semev07_tr_de, semev18_tr_de, semev19_tr_de, alm_tr_de, isear_tr_de, annot_tr_de]
    combined_master_multi_tr_de = pandas.concat(combined_master_multi_tr_de)

    combined_master_multi_tr_de = combined_master_multi_tr_de.sample(frac=1, random_state=42)
    combined_master_multi_tr_de = combined_master_multi_tr_de.reset_index(drop=True)
    combined_master_multi_tr_de.index.name = 'Id'

    combined_master_multi_tr_de.to_csv(folder.output + os.path.join("/multi_master_train_dev.csv"))

    semev07_test = pandas.read_csv(folder.raw + 'semev07_master_test.csv')
    semev18_test = pandas.read_csv(folder.raw + 'semev18_master_test.csv')
    semev19_test = pandas.read_csv(folder.raw + 'semev19_master_test.csv')
    alm_test = pandas.read_csv(folder.raw + 'alm_master_test.csv')
    isear_test = pandas.read_csv(folder.raw + 'isear_master_test.csv')
    annot_test = pandas.read_csv(folder.raw + 'annot_master_test.csv')

    combined_master_multi_test = [semev07_test, semev18_test, semev19_test, alm_test, isear_test, annot_test]
    combined_master_multi_test = pandas.concat(combined_master_multi_test)
    combined_master_multi_test = combined_master_multi_test.sample(frac=1, random_state=42)

    combined_master_multi_test = combined_master_multi_test.reset_index(drop=True)
    combined_master_multi_test.index.name = 'Id'

    combined_master_multi_test.to_csv(folder.output + os.path.join("/multi_master_test.csv"))


def generate_csv_dataset():
    parser = argparse.ArgumentParser(description='Create data_preprocessing annotation file.')
    parser.add_argument('--raw', type=str, help='the path where the raw arg_te_dev_train. mining dataset is located.',
                        default=RAW_DATASET_FOLDER)
    parser.add_argument('--output', type=str, help='the path where the generated csv should be stored.)',
                        default=OUTPUT_DATASET_FOLDER)

    args = parser.parse_args()
    generate_entire_labelset(args)


if __name__ == '__main__':
    generate_csv_dataset()
