import argparse
import csv
import datetime
import errno
import io
import logging
import os
import subprocess
import sys
import pickle

import apache_beam as beam
from apache_beam.metrics import Metrics
from apache_beam import pvalue
from apache_beam.io import WriteToText
# pylint: disable=g-import-not-at-top
# TODO(yxshi): Remove after Dataflow 0.4.5 SDK is released.
try:
    try:
        from apache_beam.options.pipeline_options import PipelineOptions
    except ImportError:
        from apache_beam.utils.pipeline_options import PipelineOptions
except ImportError:
    from apache_beam.utils.options import PipelineOptions
from PIL import Image
import tensorflow as tf

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io

from inception_resnet_v2_builder import InceptionResnetV2Builder as ModelBuilder

import numpy as np

import gcs_dataflow_utils
import sys
from tensorflow.python.framework.errors_impl import NotFoundError


reload(sys)
sys.setdefaultencoding('utf8')

slim = tf.contrib.slim

error_count = Metrics.counter('main', 'errorCount')
missing_label_count = Metrics.counter('main', 'missingLabelCount')
csv_rows_count = Metrics.counter('main', 'csvRowsCount')
labels_count = Metrics.counter('main', 'labelsCount')
labels_without_ids = Metrics.counter('main', 'labelsWithoutIds')
existing_file = Metrics.counter('main', 'existingFile')
non_existing_file = Metrics.counter('main', 'nonExistingFile')
skipped_empty_line = Metrics.counter('main', 'skippedEmptyLine')
embedding_good = Metrics.counter('main', 'embedding_good')
embedding_bad = Metrics.counter('main', 'embedding_bad')
incompatible_image = Metrics.counter('main', 'incompatible_image')
invalid_uri = Metrics.counter('main', 'invalid_file_name')
unlabeled_image = Metrics.counter('main', 'unlabeled_image')
unknown_label = Metrics.counter('main', 'unknown_label')


def gcs_file_exists(source_url):
    with gcs_dataflow_utils.open_file_read_binary(source_url) as f:
        try:
            f.read()
        except NotFoundError:
            return False
    return True


class Default(object):
    FORMAT = 'jpeg'

    # Make sure to update the default checkpoint file if using another
    # inception graph or when a newer checkpoint file is available. See
    # https://research.googleblog.com/2016/08/improving-inception-and-image.html
    IMAGE_GRAPH_CHECKPOINT_URI = (
        'gs://unipol-damage-machine-learning-retrain/unipol_retrain/nets/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt')

    #EMBEDDING_LEVELS = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
    #  'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
    #  'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'AuxLogits', 'PreLogitsFlatten']

    EMBEDDING_LEVELS = ['PreLogitsFlatten']


class ProcessCsvDoFn(beam.DoFn):


    # The try except is for compatiblity across multiple versions of the sdk
    def process(self, row):
        try:
            row = row.element
        except AttributeError:
            pass

        # Row format is: image_uri(,label_ids)*
        if not row:
            skipped_empty_line.inc()
            return

        csv_rows_count.inc()
        uri = row[0]
        if not uri or not uri.startswith('gs://'):
            invalid_uri.inc()
            return


        yield row[0]

"""
class FilterExisistingNumpyDoFn(beam.DoFn):

    def __init__(self, output_path, embedding_levels, filesystem_depth, file_name_suffix):
        self.output_path = output_path
        self.embedding_levels = embedding_levels
        self.filesystem_depth = filesystem_depth
        self.file_name_suffix = file_name_suffix

    def start_bundle(self, context=None):
        self.numpy_lists = [subprocess.check_output(["gsutil", "ls","-r", os.path.join(self.output_path, x)]).split("\n") for x in self.embedding_levels]

    def process(self, element):
        uri = element.element
        path_splits = uri.split("/");
        imagename = '/'.join(path_splits[(-1+(-1)*self.filesystem_depth):])
        output_filename = os.path.join(self.output_path,imagename+self.file_name_suffix)
        for numpy_list in self.numpy_lists:
            if output_filename not in numpy_list:
                yield uri
"""

class CheckIfOutputExistsDoFn(beam.DoFn):

    def __init__(self, output_path, embedding_levels, filesystem_depth, file_name_suffix):
        self.output_path = output_path
        self.embedding_levels = embedding_levels
        self.filesystem_depth = filesystem_depth
        self.file_name_suffix = file_name_suffix

    #def start_bundle(self, context=None):
    #    self.numpy_lists = [subprocess.check_output(["gsutil", "ls","-r", os.path.join(self.output_path, x)]).split("\n") for x in self.embedding_levels]

    def process(self, element):
        try:
            uri = element.element
        except AttributeError:
            uri = element        
        path_splits = uri.split("/");
        imagename = '/'.join(path_splits[(-1+(-1)*self.filesystem_depth):])
        exists_everywhere=True
        for emb in self.embedding_levels:
            output_filename = os.path.join(self.output_path,emb,imagename+self.file_name_suffix)
            if gcs_file_exists(output_filename):
                #yield uri
                print("output_exists", output_filename, uri)
            else:
                print("output_doesnt_exists", output_filename, uri)
                exists_everywhere=False
                break;
        if exists_everywhere:
            yield pvalue.TaggedOutput("output_exists", uri)
        else:
            yield pvalue.TaggedOutput("output_doesnt_exists", uri)

class ReadImageAndConvertToJpegDoFn(beam.DoFn):
    """Read files from GCS and convert images to JPEG format.
    We do this even for JPEG images to remove variations such as different number
    of channels.
    """

    def process(self, element):
        try:
            uri = element.element
        except AttributeError:
            uri = element

        # TF will enable 'rb' in future versions, but until then, 'r' is
        # required.
        def _open_file_read_binary(uri):
            try:
                return file_io.FileIO(uri, mode='rb')
            except errors.InvalidArgumentError:
                return file_io.FileIO(uri, mode='r')

        try:
            with _open_file_read_binary(uri) as f:
                image_bytes = f.read()
                img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # A variety of different calling libraries throw different exceptions here.
        # They all correspond to an unreadable file so we treat them equivalently.
        except Exception as e:  # pylint: disable=broad-except
            logging.exception('Error processing image %s: %s', uri, str(e))
            error_count.inc()
            return

        # Convert to desired format and output.
        output = io.BytesIO()
        img.save(output, Default.FORMAT)
        image_bytes = output.getvalue()
        yield uri, image_bytes


class EmbeddingsGraph(object):
    """Builds a graph and uses it to extract embeddings from images.
    """

    def __init__(self, tf_session,
                 embedding_levels = ['PreLogitsFlatten'],
                 trained_checkpoint_path = Default.IMAGE_GRAPH_CHECKPOINT_URI):
        self.trained_checkpoint_path = trained_checkpoint_path
        self.tf_session = tf_session
        # input_jpeg is the tensor that contains raw image bytes.
        # It is used to feed image bytes and obtain embeddings.
        self.input_jpeg, self.embeddings = self.build_graph(embedding_levels)

        init_op = tf.global_variables_initializer()
        self.tf_session.run(init_op)
        #self.restore_from_checkpoint(Default.IMAGE_GRAPH_CHECKPOINT_URI)
        self.restore_from_checkpoint(self.trained_checkpoint_path)

    def build_graph(self, embedding_levels):
        """Forms the core by building a wrapper around the inception graph.
          Here we add the necessary input & output tensors, to decode jpegs,
          serialize embeddings, restore from checkpoint etc.
          To use other Inception models modify this file. Note that to use other
          models beside Inception, you should make sure input_shape matches
          their input. Resizing or other modifications may be necessary as well.
          See tensorflow/contrib/slim/python/slim/nets/inception_v3.py for
          details about InceptionV3.
        Returns:
          input_jpeg: A tensor containing raw image bytes as the input layer.
          embedding: The embeddings tensor, that will be materialized later.
        """

        modelBuilder = ModelBuilder()
        # we are feeding a single image at a time, so batch=False
        input_jpeg, image = modelBuilder.build_input(batch=False)

        # Build Inception layers, which expect a tensor of type float from [-1, 1)
        # and shape [batch_size, height, width, channels].
        #net, end_points, ordered_end_points = modelBuilder.build_predict_model(image, final_endpoint='PreLogitsFlatten', is_training=False)
        net, end_points, ordered_end_points = modelBuilder.build_predict_model(image, final_endpoint='PreLogitsFlatten')

        endpoints = []

        for embedding_level in embedding_levels:
            endpoints.append((embedding_level,end_points[embedding_level]))

        return input_jpeg, endpoints

    def restore_from_checkpoint(self, checkpoint_path):
        vars_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=['InceptionResnetV2/Logits'])
        saver = tf.train.Saver(vars_to_restore)
        #saver.restore(self.tf_session, Default.IMAGE_GRAPH_CHECKPOINT_URI)
        #saver.restore(self.tf_session, self.trained_checkpoint_path)
        saver.restore(self.tf_session, checkpoint_path)

    def calculate_embedding(self, batch_image_bytes):
        """Get the embeddings for a given JPEG image.
        Args:
          batch_image_bytes: As if returned from [ff.read() for ff in file_list].
        Returns:
          The Inception embeddings
        """

        tensors = [e[1] for e in self.embeddings]

        embeddings_result = self.tf_session.run(
            tensors, feed_dict={self.input_jpeg: batch_image_bytes})

        results = [ (e[0], embeddings_result[i]) for i,e in enumerate(self.embeddings)]

        return results


class TFExampleFromImageDoFn(beam.DoFn):
    """Embeds image bytes and labels, stores them in tensorflow.Example.
      (uri, label_ids, image_bytes) -> (tensorflow.Example).
      Output proto contains 'label', 'image_uri' and 'embedding'.
      The 'embedding' is calculated by feeding image into input layer of image
      neural network and reading output of the bottleneck layer of the network.
      Attributes:
        image_graph_uri: an uri to gcs bucket where serialized image graph is
                         stored.
      """

    def __init__(self, embedding_levels):
        self.embedding_levels = embedding_levels
        self.tf_session = None
        self.graph = None
        self.preprocess_graph = None

    def start_bundle(self, context=None):
        # There is one tensorflow session per instance of TFExampleFromImageDoFn.
        # The same instance of session is re-used between bundles.
        # Session is closed by the destructor of Session object, which is called
        # when instance of TFExampleFromImageDoFn() is destructed.
        if not self.graph:
            self.graph = tf.Graph()
            self.tf_session = tf.InteractiveSession(graph=self.graph)
            with self.graph.as_default():
                self.preprocess_graph = EmbeddingsGraph(self.tf_session, embedding_levels)

    def process(self, element):

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        try:
            element = element.element
        except AttributeError:
            pass
        uri, image_bytes = element

        try:
            embeddings = self.preprocess_graph.calculate_embedding(image_bytes)
        except errors.InvalidArgumentError as e:
            incompatible_image.inc()
            logging.warning('Could not encode an image from %s: %s', uri, str(e))
            return

        for level_name, embedding in embeddings:
            if embedding.any():
                embedding_good.inc()
            else:
                embedding_bad.inc()
            break

        examples = {}
        for level_name, embedding in embeddings:
            feature = {
                'image_uri': _bytes_feature([uri]),
                'level_name' : _bytes_feature([level_name]),
                'embedding': _float_feature(embedding.ravel().tolist())
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            examples[level_name] = example

        yield examples


class TaggedEmbeddingsFromImageDoFn(beam.DoFn):
    """Embeds image bytes and labels, stores them in tensorflow.Example.
      (uri, label_ids, image_bytes) -> (tensorflow.Example).
      Output proto contains 'label', 'image_uri' and 'embedding'.
      The 'embedding' is calculated by feeding image into input layer of image
      neural network and reading output of the bottleneck layer of the network.
      Attributes:
        image_graph_uri: an uri to gcs bucket where serialized image graph is
                         stored.
      """

    def __init__(self, embedding_levels, trained_checkpoint_path):
        self.embedding_levels = embedding_levels
        self.trained_checkpoint_path = trained_checkpoint_path
        self.tf_session = None
        self.graph = None
        self.preprocess_graph = None

    def start_bundle(self, context=None):
        # There is one tensorflow session per instance of TFExampleFromImageDoFn.
        # The same instance of session is re-used between bundles.
        # Session is closed by the destructor of Session object, which is called
        # when instance of TFExampleFromImageDoFn() is destructed.
        if not self.graph:
            self.graph = tf.Graph()
            self.tf_session = tf.InteractiveSession(graph=self.graph)
            with self.graph.as_default():
                self.preprocess_graph = EmbeddingsGraph(self.tf_session, self.embedding_levels, self.trained_checkpoint_path)

    def process(self, element):

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        try:
            element = element.element
        except AttributeError:
            pass
        uri, image_bytes = element

        try:
            embeddings = self.preprocess_graph.calculate_embedding(image_bytes)
        except errors.InvalidArgumentError as e:
            incompatible_image.inc()
            logging.warning('Could not encode an image from %s: %s', uri, str(e))
            return

        for level_name, embedding in embeddings:
            if embedding.any():
                embedding_good.inc()
            else:
                embedding_bad.inc()
            break

        for level_name, embedding in embeddings:
            yield pvalue.TaggedOutput(level_name, (uri, embedding))

class CustomSaveToGCSasNumpyDoFn(beam.DoFn):

    def __init__(self,path,file_name_suffix, filesystem_depth=0):
        self.path = path
        self.file_name_suffix = file_name_suffix
        self.filesystem_depth = filesystem_depth

    def process(self, element):
        try:
            element = element.element
        except AttributeError:
            pass
        uri,embedding = element
        path_splits = uri.split("/");
        imagename = '/'.join(path_splits[(-1+(-1)*self.filesystem_depth):])
        output_filename = os.path.join(self.path,imagename+self.file_name_suffix)
        with gcs_dataflow_utils.open_file_write_binary(output_filename) as f:
            np.save(f,embedding)


def configure_pipeline(p, opt):
    embedding_levels = Default.EMBEDDING_LEVELS
    if opt.embedding_levels:
        print("found embedding levels:")
        embedding_levels = [ x.strip() for x in opt.embedding_levels.split(',')]
        print(embedding_levels)

    print("checking embedding levels...")
    available_levels = ModelBuilder.BASE_ENDPOINTS + ModelBuilder.HEAD_ENDPOINTS
    for lvl in embedding_levels:
        if lvl not in available_levels:
            raise ValueError(str(lvl)+' is not a level of the network.\nNetwork levels:\n'+str(embedding_levels))
    print("embedding levels are OK")

    """Specify PCollection and transformations in pipeline."""
    read_input_source = beam.io.ReadFromText(
        opt.input_path, strip_trailing_newlines=True)
    image_uris = (p
       | 'Read input' >> read_input_source
       | 'Parse input' >> beam.Map(lambda line: csv.reader([line]).next())
       | 'Parse Csv' >> beam.ParDo(ProcessCsvDoFn())
       | 'Check if output file exists' >> beam.ParDo(
           CheckIfOutputExistsDoFn(opt.output_path, embedding_levels, opt.filesystem_depth, '.npy'))
                  .with_outputs("output_exists","output_doesnt_exists")
    )
    
    if opt.logs_folder_path is not None:
        _ = (image_uris["output_exists"]
            | "encode to csv exists" >> beam.Map(lambda uri: '"'+uri+'"'+","+'"output_already_exists"')
            | 'log exists' >> WriteToText(os.path.join(opt.logs_folder_path,"already_existing_outputs.csv"))
            )
        _ = (image_uris["output_doesnt_exists"]
            | "convert to csv doesnt exists" >> beam.Map(lambda uri: '"'+uri+'"'+","+'"output_doesnt_already_exist"')
            | 'log doesnt exists' >> WriteToText(os.path.join(opt.logs_folder_path,"new_outputs.csv"))
            )
    
    tfExampleFromImage = ( image_uris["output_doesnt_exists"]
       | 'Read and convert to JPEG' >> beam.ParDo(ReadImageAndConvertToJpegDoFn())
       | 'Embed and make TFExample' >> beam.ParDo(TaggedEmbeddingsFromImageDoFn(embedding_levels, opt.trained_checkpoint_path))
                          .with_outputs(*tuple(embedding_levels))
    )
        
    


    for level_name in embedding_levels:
        _ = ( tfExampleFromImage[level_name]
           | level_name+' Save to disk' >> beam.ParDo(
               CustomSaveToGCSasNumpyDoFn(os.path.join(opt.output_path,level_name),'.npy',opt.filesystem_depth))
        )



def run(in_args=None):
    """Runs the pre-processing pipeline."""

    pipeline_options = PipelineOptions.from_dictionary(vars(in_args))
    with beam.Pipeline(options=pipeline_options) as p:
        configure_pipeline(p, in_args)


def default_args(argv):
    """Provides default values for Workflow flags."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--input_path',
      required=True,
      help='Input specified as uri to CSV file. Each line of csv file '
      'contains colon-separated GCS uri to an image and labels.')
    parser.add_argument(
      '--embedding_levels',
      default=None,
      help='A column-separated list of valid embedding levels')
    parser.add_argument(
     '--filesystem_depth',
     dest='filesystem_depth',
     default=0,
     required=True,
     type=int,
     help='Depth of the output filesystem wrt of the input filesystem.')
    parser.add_argument(
      '--output_path',
      required=True,
      help='Output directory to write results to.')
    parser.add_argument(
      '--trained_checkpoint_path',
      required=True,
      type=str,
      help='The path to the fixed network weights checkpoint'
        'that will be used for network initialization'
        'in order to perform the transfer learning'
    )
    parser.add_argument(
      '--logs_folder_path',
      default=None,
      type=str,
      help='Path of the logs files'
    )
    parser.add_argument(
      '--project',
      type=str,
      help='The cloud project name to be used for running this pipeline')

    parser.add_argument(
      '--job_name',
      type=str,
      default='generate-embeddings-numpy',
      help='A unique job identifier.')
    parser.add_argument(
      '--num_workers', default=20, type=int, help='The number of workers.')
    parser.add_argument('--cloud', default=False, action='store_true')
    parser.add_argument(
      '--worker_machine_type', default='n1-standard-1', type=str, help='The type of machines.')
    parser.add_argument(
      '--region', default='europe-west1', type=str, help='Region for dataflow workers.')
    parser.add_argument(
      '--zone', default='europe-west1-b', type=str, help='Zone for dataflow workers.')
    parser.add_argument(
      '--runner',
      help='See Dataflow runners, may be blocking'
      ' or not, on cloud or not, etc.')
    parser.add_argument(
        '--setup_file',
        default=None,
        help=
        ('Path to a setup Python file containing package dependencies. If '
         'specified, the file\'s containing folder is assumed to have the '
         'structure required for a setuptools setup package. The file must be '
         'named setup.py. More details: '
         'https://pythonhosted.org/an_example_pypi_project/setuptools.html '
         'During job submission a source distribution will be built and the '
         'worker will install the resulting package before running any custom '
         'code.'))

    parsed_args, _ = parser.parse_known_args(argv)
    parsed_args.job_name += "-"+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    if parsed_args.cloud:
        # Flags which need to be set for cloud runs.
        default_values = {
            'project':
                get_cloud_project(),
            'temp_location':
                os.path.join(os.path.dirname(parsed_args.output_path), 'temp'),
            'runner':
                'DataflowRunner',
            'save_main_session':
                True,
        }
    else:
        # Flags which need to be set for local runs.
        default_values = {
            'runner': 'DirectRunner',
        }

    for kk, vv in default_values.iteritems():
        if kk not in parsed_args or not vars(parsed_args)[kk]:
            vars(parsed_args)[kk] = vv

    return parsed_args


def get_cloud_project():
    cmd = [
      'gcloud', '-q', 'config', 'list', 'project',
      '--format=value(core.project)'
    ]
    with open(os.devnull, 'w') as dev_null:
        try:
            res = subprocess.check_output(cmd, stderr=dev_null).strip()
            if not res:
                raise Exception('--cloud specified but no Google Cloud Platform '
                                'project found.\n'
                                'Please specify your project name with the --project '
                                'flag or set a default project: '
                                'gcloud config set project YOUR_PROJECT_NAME')
            return res
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise Exception('gcloud is not installed. The Google Cloud SDK is '
                                'necessary to communicate with the Cloud ML service. '
                                'Please install and set up gcloud.')
            raise


def main(argv):
    arg_dict = default_args(argv)
    run(arg_dict)


if __name__ == '__main__':
    main(sys.argv[1:])
