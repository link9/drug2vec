"""
+word2vec with NCE loss
train and visualize the embeddings on TensorBoard
    seungbin oh

+reference
    Standford SI20
"""

import os
import pickle

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from batch_generator_mimic import batch_generator_skipgram

###

FLAGS = tf.app.flags.FLAGS
FLAGS.VOCAB_SIZE = 3268
FLAGS.BATCH_SIZE = 256
FLAGS.NUM_TRAIN_STEPS = 200000
FLAGS.SKIP_WINDOW = 10 # the context window
FLAGS.NUM_SAMPLED = 128  # Number of negative examples to sample.
FLAGS.LEARNING_RATE = 1

FLAGS.EMBED_SIZE = 128 # dimension of the word embedding vectors
FLAGS.EMBED_STEP = 2000
FLAGS.LOG_DIR = 'tensorboard/'

FLAGS.CHECKPOINTS_DIR = 'checkpoints/'
FLAGS.CHECKPOINTS_STEP = 2000

summary_dirpath = FLAGS.LOG_DIR + 'LR' + str(FLAGS.LEARNING_RATE) + 'WIN' + str(FLAGS.SKIP_WINDOW)
checkpoints_filepath = FLAGS.CHECKPOINTS_DIR + 'MIMIC-III-skipgram'

###

class SkipGramModel:
    """ Build the graph for word2vec model """
    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate, data):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.epoch = 0
        self.data = data

    def _create_placeholders(self):
        """ Step 1: define the placeholders for input and output """
        with tf.name_scope("data"):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name='center_words')
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_words')

    def _create_embedding(self):
        """ Step 2: define weights. In word2vec, it's actually the weights that we care about """
        # Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
        with tf.name_scope("embed"):
            self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size,
                                                                self.embed_size], -1.0, 1.0),
                                                                name='embed_matrix')

    def _create_loss(self):
        """ Step 3 + 4: define the model + the loss function """
        with tf.name_scope("loss"):
            # Step 3: define the inference
            embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')

            # Step 4: define loss function
            # construct variables for NCE loss
            nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                        stddev=1.0 / (self.embed_size ** 0.5)),
                                                        name='nce_weight')
            nce_bias = tf.Variable(tf.zeros([FLAGS.VOCAB_SIZE]), name='nce_bias')

            # define loss function to be NCE loss function
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                biases=nce_bias,
                                                labels=self.target_words,
                                                inputs=embed,
                                                num_sampled=self.num_sampled,
                                                num_classes=self.vocab_size), name='loss')

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                              global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """ Build the graph for our model """
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def train_model(self, batch_gen, num_train_steps):

        with tf.Session() as sess:

            # session setup
            sess.run(tf.global_variables_initializer())
            total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps

            embedding_filepath = FLAGS.LOG_DIR + 'embedding_matrix'
            # tensorboard (summary & embedding) setup
            writer = tf.summary.FileWriter(summary_dirpath, sess.graph)
            embedding_writer = tf.summary.FileWriter(FLAGS.LOG_DIR)

            # checkpoint setup
            saver = tf.train.Saver()  # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoints_filepath))
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = self.global_step.eval()


            for index in range(initial_step, initial_step + num_train_steps):
                while True:
                    try:
                        centers, targets = next(batch_gen)
                    except StopIteration:
                        self.epoch += 1
                        print('\rTrained {:d} epoch, reloading data'.format(self.epoch))
                        batch_gen = batch_generator_skipgram(self.data, "simple", FLAGS.SKIP_WINDOW,
                                                             FLAGS.BATCH_SIZE)
                        continue
                    break

                feed_dict={self.center_words: centers, self.target_words: targets}
                loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op],
                                                  feed_dict=feed_dict)
                writer.add_summary(summary, global_step=index)
                total_loss += loss_batch

                if (index + 1) % FLAGS.CHECKPOINTS_STEP == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / FLAGS.CHECKPOINTS_STEP))
                    total_loss = 0.0
                    saver.save(sess, checkpoints_filepath, index)

                if (index + 1) % FLAGS.EMBED_STEP == 0:

                    config = projector.ProjectorConfig()

                    embedding = config.embeddings.add()             # add embedding to the config file
                    embedding.tensor_name = self.embed_matrix.name
                    embedding.metadata_path = FLAGS.LOG_DIR+'int_to_drug_cd_meta.tsv' # link this tensor to its metadata file
                    # relative to LOG_DIR

                    # saves a configuration file that TensorBoard will read during startup.
                    projector.visualize_embeddings(embedding_writer, config)
                    embedding_saver = tf.train.Saver([self.embed_matrix])
                    embedding_saver.save(sess, embedding_filepath, 1) # provide global step

def deleteLog():
    import shutil
    shutil.rmtree(FLAGS.CHECKPOINTS_DIR)
    shutil.rmtree(FLAGS.LOG_DIR)

def makedir(dir_check):
    if not os.path.exists(dir_check):
        os.makedirs(dir_check)

def main():

    # reset tensorflow
    deleteLog()
    makedir('checkpoints')
    makedir('tensorboard')

    # pickle load
    path_seqs = '.. THE PATH TO PREPROCESSED PICKLE FILES ..'
    adm_visit_drug_cd = pickle.load(open(path_seqs + 'adm_visit_drug_cd.pkl', 'rb')) # nested list
    print('pickle loaded')
    print('adm_visit_drug_cd size :', len(adm_visit_drug_cd))


    model = SkipGramModel(FLAGS.VOCAB_SIZE, FLAGS.EMBED_SIZE, FLAGS.BATCH_SIZE, FLAGS.NUM_SAMPLED, FLAGS.LEARNING_RATE, adm_visit_drug_cd)
    model.build_graph()
    batch_gen = batch_generator_skipgram(adm_visit_drug_cd, "simple", FLAGS.SKIP_WINDOW, FLAGS.BATCH_SIZE)
    model.train_model(batch_gen, FLAGS.NUM_TRAIN_STEPS)

if __name__ == '__main__':
    main()