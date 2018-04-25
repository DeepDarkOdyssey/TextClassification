# Text Classification with Tensorflow

This project implements several state-of-the-art deep learning text classification algorithms using tensorflow. 

The entry of the whole project is `rnn.py`, in which I've defined the configurations that you can tune for each experiment.
Feel free to step into the source code to give you some intuitions about the whole structure.

This project uses tf.data APIs instead of placeholders, with almost no numpy functions involved, including the bucket
process when generating a batch of input.

I also provide a toy data for you to give a try.

## Requirements
```
python3 
tensorflow >= 1.7
tqdm
munch
```

## Task
Given a sentence, assign a label to it according to its content.
```
I love you.     --- BULLSHIT
```

## data & preprocess
The toy data provided in the `data/dispensatories` directory is a json file contains some Chinese dispensatories crawled
from the Internet. 

In `preprocess.py` I implements some functions to process the json file, take the key and value of 
each json dict as text and label to build the dataset, split it into train/dev/test sets, and save into the same directory.

If you want to try some other data sets, implement a new data process function and modify the code of `prepare()`
function in `run.py`

## Usage
This contains several steps:
1. Before you can get started on training the model, you need to split the raw data in to train/dev/test sets and save
them in some easy-to-load format for the the tf.data APIs. Additionally, most nlp projects need some vocabularies and 
maybe the embedding for each token, This should be done together while processing the raw data. So, run the follow script
to prepare the data once for all.
```
python run.py --prepare
```
> For other configurations like where to load and save the data and vocab and so on, check out the `run.py` for details, 
or try `python run.py -h`.

After running the prepare command, you shall find a `global_config.json` saved in current directory and some new directories 
and files has been created in the `data` directory.

2. After the dirty preprocessing jobs, you can try running an experiment with some configurations by:
```
python run.py --train model_name FastText --experiment_name test
```
This command will run a experiment with the name *test* using the *FastText* defined in the `models` directory with
default configurations. You can modify the model by changing some hyperparameters if you want.

> Note that the all available model names are defined in `models/__init__.py`. If you build a model yourself, you
must add the name to the file mentioned above.

All the data related to this experiment will be saved in `experiments/FastText/test`, including the checkpoints, 
summaries, logs, the current config and the metrics for training and evaluation, so you can check the performance of the model or reuse some config conveniently 
later.

3. When the training process in Done, you can run the following script to evaluate and make predictions on some test set.
> NOTE: You need to specify the config_path and restore_from to let the model know which weights it needs to load to 
> make predictions
```
python run.py --test --config_path XXXX --restore_from XXXX
```
4. If you need to interact with the trained model, you can run the following script to type in some words and let the 
model tell you which label it belongs to.
> NOTE: As mentioned above, you need to specify the config_path and restore from
```
python run.py --predict --config_path XXXX --restore_from XXXX
```

## Folder Structure
The architecture of this project is based on [cs230-project](https://github.com/cs230-stanford/cs230-code-examples), 
[Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) and 
[this blog](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3).
```
├── data                        - this folder contains all the data files, the raw data, the train/dev/test sets and the vocab 
│   └── dipensatories
│       ├── dev
│       ├── test
│       ├── train
│       └── vocab
├── experiments                 - this folder contains each experiment's data. 
├── models                      - this folder contains different models implemented in a single .py file.
│   ├── __init__.py
│   └── fast_text.py            - the FastText model
├── preprocess.py               - this file contains some functions to be used in process specific raw data.
├── input_fn.py                 - some functions to load processed data and build batched input using tf.data APIs.
├── run.py                      - here is the main entrance of this project.
├── utils.py                    - some utility functions.
└── vocab.py                    - here's the Vocab class .
```

## Future works

- Implements more models.
- Refine the structure.

