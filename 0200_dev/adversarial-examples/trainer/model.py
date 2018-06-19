import argparse
import logging

import tensorflow as tf
from tensorflow.contrib import layers

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils

import util
import collections
from util import override_if_not_in_args

from inception_resnet_v2_builder import InceptionResnetV2Builder as ModelBuilder

slim = tf.contrib.slim

LOGITS_TENSOR_NAME = 'logits_tensor'
IMAGE_URI_COLUMN = 'image_uri'
LABEL_COLUMN = 'label'
EMBEDDING_COLUMN = 'embedding'


class GraphMod():
    TRAIN = 1
    EVALUATE = 2
    PREDICT = 3


def build_signature(inputs, outputs):
    """Build the signature.

  Not using predic_signature_def in saved_model because it is replacing the
  tensor name, b/35900497.

  Args:
    inputs: a dictionary of tensor name to tensor
    outputs: a dictionary of tensor name to tensor
  Returns:
    The signature, a SignatureDef proto.
  """
    signature_inputs = {key: saved_model_utils.build_tensor_info(tensor)
                        for key, tensor in inputs.items()}
    signature_outputs = {key: saved_model_utils.build_tensor_info(tensor)
                         for key, tensor in outputs.items()}

    signature_def = signature_def_utils.build_signature_def(
        signature_inputs, signature_outputs,
        signature_constants.PREDICT_METHOD_NAME)

    return signature_def


def create_model():
    """Factory method that creates model to be used by generic task.py."""
    parser = argparse.ArgumentParser()
    # Label count needs to correspond to nubmer of labels in dictionary used
    # during preprocessing.
    parser.add_argument('--label_count', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--last_fixed_layer', type=str, default="PreLogitsFlatten")
    parser.add_argument(
      '--trained_checkpoint_path',
      required=True,
      type=str,
      help='The path to the fixed network weights checkpoint'
        'that will be used for network initialization'
        'in order to perform the transfer learning'
    )
    parser.add_argument('--final_layer_type', type=str, default='Softmax')
    args, task_args = parser.parse_known_args()
    override_if_not_in_args('--max_steps', '1000', task_args)
    override_if_not_in_args('--batch_size', '100', task_args)
    override_if_not_in_args('--eval_set_size', '370', task_args)
    override_if_not_in_args('--eval_interval_secs', '2', task_args)
    override_if_not_in_args('--log_interval_secs', '2', task_args)
    override_if_not_in_args('--min_train_eval_rate', '2', task_args)
    override_if_not_in_args('--final_layer_type', 'Softmax', task_args)
    #override_if_not_in_args('--last_fixed_layer', 'PreLogitsFlatten', task_args)
    return Model(args.label_count, args.dropout, last_fixed_layer=args.last_fixed_layer, trained_checkpoint_path=args.trained_checkpoint_path, final_layer_type=args.final_layer_type), task_args


class GraphReferences(object):
    """Holder of base tensors used for training model using common task."""

    def __init__(self):
        self.examples = None
        self.train = None
        self.global_step = None
        self.metric_updates = []
        self.metric_values = []
        self.keys = None
        self.predictions = []
        self.input_jpeg = None


class Model(object):
    
    
    def __init__(self, label_count, dropout, 
                 last_fixed_layer = "PreLogitsFlatten", 
                 trained_checkpoint_path=ModelBuilder.IMAGE_GRAPH_CHECKPOINT_URIS[0],
                 final_layer_type="Softmax"):
        self.last_fixed_layer = last_fixed_layer
        self.label_count = label_count
        self.dropout = dropout
        self.trained_checkpoint_path = trained_checkpoint_path
        self.final_layer_type = final_layer_type

    
    def add_final_training_ops(self,
                             embeddings,
                             all_labels_count,
                             hidden_layer_size=None,
                             dropout_keep_prob=None):
        
        modelBuilder = ModelBuilder()
        
        last_fixed_layer = self.last_fixed_layer
        
        with tf.name_scope('final_ops'):
            if dropout_keep_prob:
                final, endpoints, ordered_endpoints = modelBuilder.build_train_model(embeddings, 
                                                                                 num_classes=all_labels_count,
                                                                                 is_training=True,
                                                                                 dropout_keep_prob=dropout_keep_prob,
                                                                                 final_endpoint=last_fixed_layer,
                                                                                 final_layer_type=self.final_layer_type,
                                                                                 reverse=True)
            else:
                final, endpoints, ordered_endpoints = modelBuilder.build_predict_model(embeddings, 
                                                                                 num_classes=all_labels_count,
                                                                                 #is_training=is_training,
                                                                                 #dropout_keep_prob=self.dropout if is_training else None,
                                                                                 final_endpoint=last_fixed_layer,
                                                                                 final_layer_type=self.final_layer_type,
                                                                                 reverse=True)
        logits = endpoints[ModelBuilder.LOGITS_ENDPOINT]
        return final, logits
        
    
    def build_inception_graph(self):
        embedding_layer_name = self.last_fixed_layer
        
        modelBuilder = ModelBuilder()
        
        image_str_tensor, image = modelBuilder.build_input()

        inception_input = image
        
        _, end_points, _ = modelBuilder.build_predict_model(image, 
                                            final_endpoint=embedding_layer_name,
                                            final_layer_type=self.final_layer_type,
                                            reverse=False)
        

        inception_embeddings = end_points[embedding_layer_name]
        
        return image_str_tensor, inception_embeddings
        

    def build_graph(self, data_paths, batch_size, graph_mod):
        modelBuilder = ModelBuilder()
        last_fixed_layer = self.last_fixed_layer
        
        
        """Builds generic graph for training or eval."""
        tensors = GraphReferences()
        is_training = graph_mod == GraphMod.TRAIN
        if data_paths:
            tensors.keys, tensors.examples = util.read_examples(
              data_paths,
              batch_size,
              shuffle=is_training,
              num_epochs=None if is_training else 2)
        else:
            tensors.examples = tf.placeholder(tf.string, name='input', shape=(None,))

        if graph_mod == GraphMod.PREDICT:
            inception_input, inception_embeddings = self.build_inception_graph()
            
            #image_str_tensor, net_input = modelBuilder.build_input()
            #last_layer, end_points, ordered_end_points = modelBuilder.build_predict_model(net_input, final_endpoint=last_fixed_layer)
            #inception_input = image_str_tensor
            #inception_embeddings = last_layer
            
            
            
            # Build the Inception graph. We later add final training layers
            # to this graph. This is currently used only for prediction.
            # For training, we use pre-processed data, so it is not needed.
            embeddings = inception_embeddings
            tensors.input_jpeg = inception_input
        else:
            embeddings_shape = ModelBuilder.DEFAULT_LAYER_SHAPES[self.last_fixed_layer][1:]
            
            # For training and evaluation we assume data is preprocessed, so the
            # inputs are tf-examples.
            # Generate placeholders for examples.
            with tf.name_scope('inputs'):
                feature_map = {
                    'image_uri':
                        tf.FixedLenFeature(
                            shape=[], dtype=tf.string, default_value=['']),
                    # Some images may have no labels. For those, we assume a default
                    # label. So the number of labels is label_count+1 for the default
                    # label.
                    'label':
                        tf.FixedLenFeature(
                            shape=[1], dtype=tf.int64,
                            default_value=[self.label_count]),
                    'embedding':
                        tf.FixedLenFeature(
                            #shape=[self.embeddings_graph.settings.embedding_tensor_size], 
                            shape=embeddings_shape,
                            dtype=tf.float32)
                }
            parsed = tf.parse_example(tensors.examples, features=feature_map)
            labels = tf.squeeze(parsed['label'])
            uris = tf.squeeze(parsed['image_uri'])
            embeddings = parsed['embedding']

        # We assume a default label, so the total number of labels is equal to
        # label_count+1.
        all_labels_count = self.label_count + 1
        #print "all_labels_count", all_labels_count
        with tf.name_scope('final_ops'):
            final, logits = self.add_final_training_ops(
                                  embeddings,
                                  all_labels_count,
                                  dropout_keep_prob=self.dropout if is_training else None)

        # Prediction is the index of the label with the highest score. We are
        # interested only in the top score.
        # Or in every result if final_layer_type == Sigmoid
        prediction = tf.argmax(final, 1)

        tensors.predictions = [prediction, final, embeddings]

        if graph_mod == GraphMod.PREDICT:
            return tensors

        with tf.name_scope('evaluate'):
            loss_value = loss(logits, labels, self.final_layer_type)
        # Add to the Graph the Ops that calculate and apply gradients.
        if is_training:
            tensors.train, tensors.global_step = training(loss_value)
        else:
            tensors.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Add means across all batches.
        
        loss_updates, loss_op = util.loss(loss_value)
        if self.final_layer_type == 'Softmax':
            #accuracy_updates, accuracy_op = util.accuracy(logits, labels)
            accuracy_updates, accuracy_op = util.intersection_over_union_softmax(logits, labels, all_labels_count)
        elif self.final_layer_type == 'Sigmoid':
            #accuracy_updates, accuracy_op = util.accuracy(logits, labels)
            accuracy_updates, accuracy_op = util.intersection_over_union_sigmoid(logits, labels, all_labels_count)
        else:
            raise ValueError("Softmax or Sigmoid!!!!!!")

        if not is_training:
            tf.summary.scalar('accuracy', accuracy_op)
            tf.summary.scalar('loss', loss_op)

        tensors.metric_updates = loss_updates + accuracy_updates
        tensors.metric_values = [loss_op, accuracy_op]
        return tensors

    def build_train_graph(self, data_paths, batch_size):
        return self.build_graph(data_paths, batch_size, GraphMod.TRAIN)

    def build_eval_graph(self, data_paths, batch_size):
        return self.build_graph(data_paths, batch_size, GraphMod.EVALUATE)


        
    def restore_from_checkpoint(self, session, trained_checkpoint_file):
        """To restore model variables from the checkpoint file.

           The graph is assumed to consist of an inception model and other
           layers including a softmax and a fully connected layer. The former is
           pre-trained and the latter is trained using the pre-processed data. So
           we restore this from two checkpoint files.
        Args:
          session: The session to be used for restoring from checkpoint.
          inception_checkpoint_file: Path to the checkpoint file for the Inception
                                     graph.
          trained_checkpoint_file: path to the trained checkpoint for the other
                                   layers.
        """
        
        #checkpoint_path = ModelBuilder.IMAGE_GRAPH_CHECKPOINT_URIS[0]
        checkpoint_path = self.trained_checkpoint_path
        
        # Get all variables to restore. Exclude Logits and AuxLogits because they
        # depend on the input data and we do not need to intialize them from
        # checkpoint.
        variables_to_exclude_from_scope = ModelBuilder.CHECKPOINT_VARIABLES_TO_EXCLUDE_FROM_SCOPE
        all_vars = tf.contrib.slim.get_variables_to_restore(
            exclude=variables_to_exclude_from_scope
        )
        
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        # Remove variables that do not exist in the inception checkpoint (for
        # example the final softmax and fully-connected layers).
        inception_vars = {
            var.op.name: var
            for var in all_vars if var.op.name in var_to_shape_map
        }

        saver = tf.train.Saver(inception_vars)
        saver.restore(session, checkpoint_path)
        
        
        

        # Restore the rest of the variables from the trained checkpoint.
        #exclude_scopes = self.embeddings_graph.settings.variables_to_exclude_from_scope
        exclude_scopes = variables_to_exclude_from_scope
        #embeddings_model_checkpoint_file = self.embeddings_graph.settings.image_graph_checkpoint_uri
        embeddings_model_checkpoint_file = checkpoint_path
        
        embeddings_model_all_vars = tf.contrib.slim.get_variables_to_restore(exclude=exclude_scopes)
        reader = tf.train.NewCheckpointReader(embeddings_model_checkpoint_file)
        var_to_shape_map = reader.get_variable_to_shape_map()
        embeddings_model_vars = {
            var.op.name: var
            for var in embeddings_model_all_vars if var.op.name in var_to_shape_map
        }
        
        trained_vars = tf.contrib.slim.get_variables_to_restore(
            exclude=exclude_scopes + embeddings_model_vars.keys())
            #exclude=inception_exclude_scopes + inception_vars.keys())
        trained_saver = tf.train.Saver(trained_vars)
        trained_saver.restore(session, trained_checkpoint_file)
        

    def build_prediction_graph(self):
        """Builds prediction graph and registers appropriate endpoints."""

        tensors = self.build_graph(None, 1, GraphMod.PREDICT)

        keys_placeholder = tf.placeholder(tf.string, shape=[None])
        inputs = {
            'key': keys_placeholder,
            'image_bytes': tensors.input_jpeg
        }

        # To extract the id, we need to add the identity function.
        keys = tf.identity(keys_placeholder)
        outputs = {
            'key': keys,
            'prediction': tensors.predictions[0],
            'scores': tensors.predictions[1]
        }

        return inputs, outputs

    def export(self, last_checkpoint, output_dir):
        """Builds a prediction graph and xports the model.

        Args:
          last_checkpoint: Path to the latest checkpoint file from training.
          output_dir: Path to the folder to be used to output the model.
        """
        logging.info('Exporting prediction graph to %s', output_dir)
        with tf.Session(graph=tf.Graph()) as sess:
            # Build and save prediction meta graph and trained variable values.
            inputs, outputs = self.build_prediction_graph()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            self.restore_from_checkpoint(sess, last_checkpoint)
            signature_def = build_signature(inputs=inputs, outputs=outputs)
            signature_def_map = {
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
            }
            builder = saved_model_builder.SavedModelBuilder(output_dir)
            builder.add_meta_graph_and_variables(
              sess,
              tags=[tag_constants.SERVING],
              signature_def_map=signature_def_map)
            builder.save()

    def format_metric_values(self, metric_values):
        """Formats metric values - used for logging purpose."""
        
        
        # Early in training, metric_values may actually be None.
        loss_str = 'N/A'
        accuracy_str = 'N/A'
        try:
            loss_str = '%.3f' % metric_values[0]
            if isinstance(metric_values[1], collections.Iterable):
                accuracy_str = ' '.join('%.3f' % m for m in metric_values[1])
            else:
                accuracy_str = '%.3f' % metric_values[1]
        except (TypeError, IndexError):
            pass

        return '%s, %s' % (loss_str, accuracy_str)


def loss(logits, labels, final_layer_type='Softmax'): #or Sigmoid
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].
    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    #print final_layer_type, 'Calculating loss'
    if final_layer_type=='Softmax':
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                          logits=logits, labels=labels, name='xentropy')
    elif final_layer_type == 'Sigmoid':
        n_classes = logits.shape[1]
        one_hotted = tf.one_hot(labels, n_classes)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=one_hotted, name='xentropy')
    else:
        raise ValueError("Softmax or Sigmoid!!!!!!")
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss_op):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(epsilon=0.001)
        train_op = optimizer.minimize(loss_op, global_step)
        return train_op, global_step
