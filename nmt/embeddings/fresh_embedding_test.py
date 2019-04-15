import tensorflow as tf
import numpy as np

from nmt.embeddings.fresh_embedding import FreshEmbedding
from nmt import misc_utils


class FreshEmbeddingTest(tf.test.TestCase):

    def testFreshEmbedding(self):
        vocab_file = misc_utils.get_test_data('iwslt15.vocab.100.en')
        embedder = FreshEmbedding(vocab_file=vocab_file)
        inputs = np.array([
            ['I', 'am', 'a', 'test']
        ])
        inputs = tf.constant(inputs,dtype=tf.string)
        length = np.array([4])
        length = tf.constant(length,dtype=tf.int32)
        params = {
            'batch_size': 1
        }
        embedded = embedder.embedding(inputs, length, params)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            embedded = sess.run(embedded)
            print(embedded)


if __name__ == '__main__':
    tf.test.main()
