#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import List, Dict
import csv 
from datasets import load_dataset

logger = logging.getLogger(__name__)

MAP_LABEL_TRANSLATION = {
      0: 'negative',
      1: 'neutral',
      2: 'positive'
}


def save_as_translations(original_save_path: Path, data_to_save: List[Dict]) -> None:
    file_name = 's2s-' + original_save_path.name
    file_path = original_save_path.parent / file_name

    print(f'Saving into: {file_path}')
    with open(file_path, 'wt') as f_write:
        for data_line in data_to_save:
            label = data_line['label']
            new_label = MAP_LABEL_TRANSLATION[label]
            data_line['label'] = new_label
            data_line_str = json.dumps(data_line)
            f_write.write(f'{data_line_str}\n')


def main() -> None:
    loaded_data = load_dataset('arize-ai/beer_reviews_label_drift_neg')
    logger.info(f'Loaded dataset imdb: {loaded_data}')
    loaded_data['training'] = loaded_data['training'].to_json('training.json')
    loaded_data['validation'] = loaded_data['validation'].to_json('validation.json')
    loaded_data['production'] = loaded_data['production'].to_json('production.json')
    save_path = Path('data/')
    save_train_path = save_path / 'training.json'
    save_valid_path = save_path / 'validation.json'
    save_test_path = save_path / 'production.json'
    if not save_path.exists():
        save_path.mkdir()

    # Read train and validation data
    data_train, data_valid, data_test = [], [], []
    for source_data, dataset, max_size in [
        (loaded_data['training'], data_train, None),
        (loaded_data['production'], data_valid, None)
    ]:
        for i, data in enumerate(source_data):
            if max_size is not None and i >= max_size:
                break
            data_line = {
                'label': int(data['label']),
                'text': data['text'],
            }
            dataset.append(data_line)
    logger.info(f'Train: {len(data_train):6d}')

    # Split validation set into 2 classes for validation and test splitting
    data_class_1, data_class_2,data_class_3= [], [],[]
    for data in data_valid:
        label = data['label']
        if label == 0:
            data_class_1.append(data)
        elif label == 1:
            data_class_2.append(data)
        elif label == 2:
            data_class_3.append(data)
    logger.info(f'Label 1: {len(data_class_1):6d}')
    logger.info(f'Label 2: {len(data_class_2):6d}')
    logger.info(f'Label 3: {len(data_class_3):6d}')

    # Split 2 classes into validation and test
    size_half_class_1 = int(len(data_class_1) / 3)
    size_half_class_2 = int(len(data_class_2) / 3)
    size_half_class_3 = int(len(data_class_3) / 3)
    data_valid = data_class_1[:size_half_class_1] + data_class_2[:size_half_class_2]+data_class_3[:size_half_class_3]
    data_test = data_class_1[size_half_class_1:] + data_class_2[size_half_class_2:]+data_class_3[:size_half_class_3]
    logger.info(f'Valid: {len(data_valid):6d}')
    logger.info(f'Test : {len(data_test):6d}')

    # Save files
    for file_path, data_to_save in [
        (save_train_path, data_train),
        (save_valid_path, data_valid),
        (save_test_path, data_test)
    ]:
        print(f'Saving into: {file_path}')
        with open(file_path, 'wt') as f_write:
            for data_line in data_to_save:
                data_line_str = json.dumps(data_line)
                f_write.write(f'{data_line_str}\n')

        save_as_translations(file_path, data_to_save)


if __name__ == '__main__':
    
    main()
