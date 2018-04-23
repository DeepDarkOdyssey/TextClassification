import argparse
import os
import json
from munch import Munch
from preprocess import dispensatories_preprocess
from vocab import Vocab
from input_fn import build_dataset, build_inputs, build_predict_dataset
from models import build_model
from utils import prepare_experiment, set_logger, load_config


parser = argparse.ArgumentParser()

# which action to take
parser.add_argument('--prepare', action='store_true',
                    help='create directories, process the raw data and build vocabularies')
parser.add_argument('--train', action='store_true',
                    help='train the model')
parser.add_argument('--test', action='store_true',
                    help='train the model')
parser.add_argument('--predict', action='store_true',
                    help='predict with raw input')

# experiment configuration
parser.add_argument('--logger_name', type=str, default='test',
                    help='logger name')
parser.add_argument('--experiments_dir', type=str, default='experiments/',
                    help='the directory to store all the experiments data')
parser.add_argument('--model_name', type=str, default='FastText',
                    help='the name of the model')
parser.add_argument('--experiment_name', type=str, default='test',
                    help='the unique name of this experiment')
parser.add_argument('--config_path', type=str, default='',
                    help='path to load config that has been saved before, if specified')
parser.add_argument('--save_result', type=bool, default=False,
                    help='whether to save the predictions during evaluation')
parser.add_argument('--result_name', type=str, default='',
                    help='the result file which the results generated during testing will be stored in')
parser.add_argument('--fixed_configs', nargs='+', type=str,
                    default=['config_path', 'save_result', 'result_name', 'restore_from'],
                    help='configs need to be fixed when loading another config')

# model hyperparameters
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size of the dataset')
parser.add_argument('--num_parallel_calls', type=int, default=8,
                    help='how many threads to be used in dataset')
parser.add_argument('--embed_size', type=int, default=150,
                    help='the dimensions of embedding vectors')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='the learning rate of the optimizer')
parser.add_argument('--keep_prob', type=float, default=0.5,
                    help='the keep probabilities for dropout')

# preparing settings
parser.add_argument('--vocab_size', type=int, default=0,
                    help='how many tokens you want to maintain in your vocab')
parser.add_argument('--data_path', type=str, default='data/dispensatories/药品说明书数据.json',
                    help='the path of the data file')
parser.add_argument('--data_dir', type=str, default='data/dispensatories',
                    help='the directory to save the processed data')
parser.add_argument('--vocab_dir', type=str, default='data/dispensatories/vocab',
                    help='the directory to store the built vocab')
parser.add_argument('--global_config', type=str, default='global_config.json',
                    help='some configurations derived from the preparation')

# training settings
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs to train the model on training set')
parser.add_argument('--restore_from', type=str, default='',
                    help="the checkpoint file or the directory it's stored in")
parser.add_argument('--log_freq', type=int, default=10,
                    help='the frequency of logging')
parser.add_argument('--summary_freq', type=int, default=10,
                    help='the frequency of saving a summary')


def prepare(config):
    # make sure every directory in config exists
    for key, value in vars(config).items():
        if key.endswith('dir') and not os.path.exists(value):
            os.mkdir(value)

    print('Processing the raw data...')
    train_size, dev_size, test_size = dispensatories_preprocess(config.data_path, config.data_dir)

    print('Building vocabularies...')
    char_vocab = Vocab([os.path.join(config.data_dir, 'train', 'texts.txt')], sep=' ')
    char_vocab.filter_by_count(2)
    label_vocab = Vocab([os.path.join(config.data_dir, 'train', 'labels.txt')], sep=' ', use_special_token=False)

    char_vocab.save_to(os.path.join(config.vocab_dir, 'char_vocab.data'))
    label_vocab.save_to(os.path.join(config.vocab_dir, 'label_vocab.data'))
    print('Saving vocab to {}'.format(config.vocab_dir))

    global_config = {
        'train_size': train_size,
        'dev_size': dev_size,
        'test_size': test_size,
        'char_vocab_size': char_vocab.size,
        'label_vocab_size': label_vocab.size
    }
    with open(config.global_config, 'w') as f:
        json.dump(global_config, f, ensure_ascii=False, indent=4)
    print('Saving global config to {}'.format(config.global_config))


def train(config):
    # with open(config.global_config) as f:
    #     global_config = json.load(f)
    # config.update(global_config)
    config = load_config(config, config.global_config)

    config = prepare_experiment(config)

    # save current config to the experiment directory
    with open(os.path.join(config.exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    set_logger(config)

    char_vocab = Vocab()
    char_vocab.load_from(os.path.join(config.vocab_dir, 'char_vocab.data'))
    label_vocab = Vocab(use_special_token=False)
    label_vocab.load_from(os.path.join(config.vocab_dir, 'label_vocab.data'))

    train_set = build_dataset(config, 'train', char_vocab, label_vocab)
    dev_set = build_dataset(config, 'dev', char_vocab, label_vocab)

    inputs = build_inputs(train_set.output_types, train_set.output_shapes)
    model = build_model(config, inputs)
    model.train(train_set, dev_set)


def test(config):
    if not config.config_path or not config.restore_from:
        raise AttributeError('You need to specify config_path and restore_from')
    else:
        config = load_config(config, config.config_path)

    set_logger(config)

    char_vocab = Vocab()
    char_vocab.load_from(os.path.join(config.vocab_dir, 'char_vocab.data'))
    label_vocab = Vocab(use_special_token=False)
    label_vocab.load_from(os.path.join(config.vocab_dir, 'label_vocab.data'))

    test_set = build_dataset(config, 'test', char_vocab, label_vocab)
    inputs = build_inputs(test_set.output_types, test_set.output_shapes)

    model = build_model(config, inputs)
    eval_metrics, results = model.evaluate(test_set)

    print('Eval metrics: {}'.format(eval_metrics))
    if config.result_name:
        with open(os.path.join(config.result_dir, os.path.join(config.result_name)) + '.json', 'w') as f:
            json.dump(eval_metrics, f, indent=4)

        with open(os.path.join(config.result_dir, os.path.join(config.result_name)) + '.txt', 'w') as f:
            for result in results:
                f.write(label_vocab.id2token[result] + '\n')


def predict(config):
    if not config.config_path:
        raise AttributeError('You need to specify config_path, which the model can load from.')
    else:
        config = load_config(config, config.config_path)

    char_vocab = Vocab()
    char_vocab.load_from(os.path.join(config.vocab_dir, 'char_vocab.data'))
    label_vocab = Vocab(use_special_token=False)
    label_vocab.load_from(os.path.join(config.vocab_dir, 'label_vocab.data'))

    text = input('请输入文本：')
    predict_set = build_predict_dataset(' '.join(text), char_vocab)
    inputs = build_inputs(predict_set.output_types, predict_set.output_shapes)

    model = build_model(config, inputs)
    _, results = model.evaluate(predict_set)
    print('分类结果： {}'.format(label_vocab.id2token[results[0]]))


def run():
    args = parser.parse_args()
    config = Munch().fromDict(vars(args))
    print(config)
    if config.prepare:
        prepare(config)
    if config.train:
        train(config)
    if config.test:
        test(config)
    if config.predict:
        predict(config)


if __name__ == '__main__':
    run()