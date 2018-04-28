"""
This module implements a Vocab class for converting token to id and back
"""

from collections import Counter
import numpy as np


class Vocab(object):
    def __init__(self, fins=[], sep=None, use_special_token=True, unk_token='<UNK>', pad_token='<PAD>'):
        self.token2id = {}
        self.id2token = []
        self.counter = Counter()

        if use_special_token:
            self.unk_token = unk_token
            self.pad_token = pad_token
            self.special_tokens = [self.pad_token, self.unk_token]
        else:
            self.special_tokens = []

        if fins:
            for f in fins:
                self.read_tokens_from_file(f, sep)
        self.embeddings = None

    def read_tokens_from_file(self, file_path, sep=None):
        for line in open(file_path, encoding='utf8'):
            if sep:
                self.counter.update(line.strip().split(sep=sep))
            else:
                self.counter.update(line.strip())
        self.update_vocab()

    def add_tokens(self, tokens=[]):
        self.counter.update(tokens)

    def update_vocab(self):
        self.id2token = self.special_tokens + [token for token, count in self.counter.most_common()]
        self.token2id = dict([(token, i) for i, token in enumerate(self.id2token)])

    @property
    def size(self):
        return len(self.id2token)

    def filter_by_count(self, min_cnt=1):
        self.id2token = self.special_tokens + [token for token, count in self.counter.most_common() if count >= min_cnt]
        self.token2id = dict([(token, i) for i, token in enumerate(self.id2token)])

    def filter_by_size(self, max_size=30000):
        self.id2token = self.special_tokens + [token for token, count in self.counter.most_common(max_size)]
        self.token2id = dict([(token, i) for i, token in enumerate(self.id2token)])

    def load_pretrained_embeddings(self, embedding_path):
        """
        loads the pretrained embeddings from embedding_path,
        tokens not in pretrained embeddings will be filtered
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            for line in fin:
                contents = line.strip().split()
                token = contents[0]
                if token not in self.token2id:
                    continue
                trained_embeddings[token] = list(map(float, contents[1:]))
                embed_size = len(contents) - 1
        # load embeddings
        self.embeddings = np.random.randn([self.size, embed_size])
        for token in self.id2token:
            if token in trained_embeddings:
                self.embeddings[self.token2id[token]] = trained_embeddings[token]

    def save_to(self, save_path):
        with open(save_path, 'w', encoding='utf8') as f:
            for token in self.id2token:
                f.write('{} {}\n'.format(token, self.counter[token]))

    def load_from(self, load_path):
        tokens = []
        with open(load_path, encoding='utf8') as f:
            for line in f.readlines():
                token, count = line[:-1].split()
                self.counter[token] = count
                tokens.append(token)
        self.id2token = tokens
        self.token2id = dict([(token, i) for i, token in enumerate(self.id2token)])

