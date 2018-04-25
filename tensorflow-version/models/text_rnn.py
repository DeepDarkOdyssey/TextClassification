import tensorflow as tf
import tqdm
import logging
import os
import json
from models.model_utils import rnn


class TextRNN:
    def __init__(self, config, inputs):
        self.config = config

        self.logger = logging.getLogger(config.logger_name)

        self.iterator = inputs.iterator
        self.texts = inputs.texts
        self.text_lens = inputs.text_lens
        self.labels = inputs.labels

        self._build_graph()

    def _add_placeholders(self):
        self.keep_prob = tf.placeholder(tf.float32)

    def _embed(self):
        with tf.device('/cpu:0'):
            self.embeddings = tf.get_variable('embeddings',
                                              shape=[self.config.char_vocab_size, self.config.embed_size])

            self.text_emb = tf.nn.embedding_lookup(self.embeddings, self.texts)

    def _inference(self):
        _, text_vector = rnn(self.config.rnn_mode, self.config.cell_type, self.text_emb, self.text_lens,
                             self.config.hidden_size, self.config.num_layers, self.config.keep_prob)
        self.logits = tf.layers.dense(text_vector, self.config.label_vocab_size)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.labels), tf.float32))

    def _compute_loss(self):
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        )

    def _add_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss, self.global_step)

    def _add_metrics(self):
        with tf.variable_scope('metrics'):
            self.metrics = {
                'accuracy': tf.metrics.accuracy(labels=self.labels, predictions=self.predictions),
                'loss': tf.metrics.mean(self.loss)
            }
        self.metrics_update_op = tf.group(*[update_op for _, update_op in self.metrics.values()])
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, 'metrics')
        self.metrics_reset_op = tf.variables_initializer(metric_variables)

    def _add_summaries(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summary_op = tf.summary.merge_all()

    def _build_graph(self):
        self._add_placeholders()
        self._embed()
        self._inference()
        self._compute_loss()
        self._add_train_op()
        self._add_metrics()
        self._add_summaries()

    def save_metrics(self, metrics, file_name):
        with open(os.path.join(self.config.exp_dir, file_name) + '.json', 'w', encoding='utf8') as f:
            metrics = {k: float(v) for k, v in metrics.items()}
            json.dump(metrics, f, ensure_ascii=False)

    def train_epoch(self, sess, epoch, dataset, writer, log_freq, summary_freq):
        sess.run(self.iterator.make_initializer(dataset))
        # TODO: will this add redundant ops to the graph? I'm not sure

        sess.run(self.metrics_reset_op)
        step = 0
        t = tqdm.trange(self.config.train_size // self.config.batch_size + 1, desc='epoch {}'.format(epoch))
        while True:
            try:
                _, __, loss, accuracy, summaries, global_step = sess.run(
                    [self.train_op, self.metrics_update_op, self.loss, self.accuracy, self.summary_op,
                     self.global_step],
                    feed_dict={self.keep_prob: self.config.keep_prob}
                )
                step += 1

            except tf.errors.OutOfRangeError:
                metrics_values = {k: v[0] for k, v in self.metrics.items()}
                metrics_val = sess.run(metrics_values)
                sess.run(self.metrics_reset_op)
                t.update(self.config.train_size % log_freq)
                t.close()
                break

            if step % log_freq == 0:
                self.logger.info('Train loss {:.4f}, accuracy {:.4%}'.format(loss, accuracy))
                t.update(log_freq)
                t.set_postfix_str('loss: {:.4f}, acc: {:.2%}'.format(loss, accuracy))

            if summary_freq > 0 and step % summary_freq == 0:
                writer.add_summary(summaries, global_step)

        self.logger.info('Train metrics: {}'.format(metrics_val))
        return metrics_val

    def train(self, train_set, dev_set):
        last_saver = tf.train.Saver()
        best_saver = tf.train.Saver(max_to_keep=1)

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.tables_initializer())

            if self.config.restore_from:
                load_saver = tf.train.Saver()
                if os.path.isdir(self.config.restore_from):
                    save_path = tf.train.latest_checkpoint(self.config.restore_from)
                else:
                    save_path = self.config.restore_from
                load_saver.restore(sess, save_path)

            else:
                sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, 'train'), graph=sess.graph)
            if dev_set:
                eval_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, 'eval'), graph=sess.graph)
            else:
                eval_writer = None

            self.logger.info('Start training ...')
            best_eval_acc = 0
            for epoch in range(1, self.config.num_epochs + 1):
                self.logger.info('Epoch {}/{}'.format(epoch, self.config.num_epochs))
                train_metrics = self.train_epoch(sess, epoch, train_set, train_writer, self.config.log_freq,
                                                 self.config.summary_freq)
                self.save_metrics(train_metrics, 'last_train_metrics')
                last_save_path = os.path.join(self.config.ckpt_dir, 'last_weights', 'after_epoch')
                last_saver.save(sess, save_path=last_save_path, global_step=epoch)

                if dev_set:
                    eval_metrics, results = self.eval_epoch(sess, dev_set, eval_writer)
                    self.logger.info('Eval metrics: {}'.format(eval_metrics))
                    eval_acc = eval_metrics['accuracy']
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        best_save_path = os.path.join(self.config.ckpt_dir, 'best_weights', 'after_epoch')
                        save_path = best_saver.save(sess, best_save_path, global_step=epoch)
                        self.logger.info(
                            '- Found new best accuracy, best weights has been saved in {}'.format(save_path)
                        )
                        self.save_metrics(eval_metrics, 'best_eval_metrics')

    def eval_epoch(self, sess, eval_set, writer=None):
        sess.run(self.iterator.make_initializer(eval_set))
        results = []

        while True:
            try:
                predictions, _ = sess.run([self.predictions, self.metrics_update_op], feed_dict={self.keep_prob: 1.0})
                if self.config.save_result:
                    results.extend(predictions.tolist())

            except tf.errors.OutOfRangeError:
                metrics_values = {k: v[0] for k, v in self.metrics.items()}
                metrics_val = sess.run(metrics_values)
                sess.run(self.metrics_reset_op)
                break

        if writer:
            global_step = sess.run(self.global_step)
            for k, v in metrics_val.items():
                summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v)])
                writer.add_summary(summary, global_step=global_step)

        return metrics_val, results

    def evaluate(self, eval_set):
        saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.tables_initializer())
            sess.run(self.metrics_reset_op)
            load_path = self.config.restore_from
            if os.path.isdir(load_path):
                load_path = tf.train.latest_checkpoint(load_path)
            saver.restore(sess, load_path)

            eval_metrics, results = self.eval_epoch(sess, eval_set)

        return eval_metrics, results


