import json
import os
from sklearn.model_selection import train_test_split


def dispensatories_preprocess(data_path, data_dir):
    """Load and process the dispensatory json files, and split it into train/dev/test sets"""
    with open(data_path, encoding='utf8') as f:
        data = json.load(f)

    texts, labels = [], []
    for d in data:
        for key, value in d.items():
            if key != 'url' and len(value) > 1 and value != 'None':
                texts.append(' '.join(value.strip().replace('\n', '').replace('\r', '')))
                labels.append(key.strip())

    assert len(texts) == len(labels)

    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels,
                                                                          test_size=10000, random_state=20180419)
    train_texts, dev_texts, train_labels, dev_labels = train_test_split(train_texts, train_labels,
                                                                        test_size=10000, random_state=20180419)

    save_data(train_texts, train_labels, data_dir, 'train')
    save_data(dev_texts, dev_labels, data_dir, 'dev')
    save_data(test_texts, test_labels, data_dir, 'test')

    return len(train_labels), len(dev_labels), len(test_labels)


def save_data(texts, labels, save_dir, mode):
    if not os.path.exists(os.path.join(save_dir, mode)):
        os.mkdir(os.path.join(save_dir, mode))

    with open(os.path.join(save_dir, mode, 'texts.txt'), 'w', encoding='utf8') as f:
        for text in texts:
            f.write(text + '\n')
    with open(os.path.join(save_dir, mode, 'labels.txt'), 'w', encoding='utf8') as f:
        for label in labels:
            f.write(label + '\n')

