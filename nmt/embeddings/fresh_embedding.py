import tensorflow as tf

from nmt.embeddings.base_embedding import BaseEmbedding


class FreshEmbedding(BaseEmbedding):

    def _embedding(self, inputs, length, params):
        if params['num_partitions'] <= 1:
            partitioner = None
        else:
            partitioner = tf.fixed_size_partitioner(num_shards=params['num_partitions'])

        with tf.variable_scope(self.scope, partitioner=partitioner, reuse=tf.AUTO_REUSE) as scope:
            embedding = tf.get_variable(
                name=self.name,
                shape=[params['vocab_size'], params['embedding_size']],
                dtype=self.dtype)

            inputs_ids = self.str2id.lookup(inputs)
            embedded_inputs = tf.nn.embedding_lookup(embedding, inputs_ids)

        return embedded_inputs
