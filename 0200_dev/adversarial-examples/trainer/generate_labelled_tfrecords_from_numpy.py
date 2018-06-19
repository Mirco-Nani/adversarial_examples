import argparse
import csv
import datetime
import errno
import io
import logging
import os
import subprocess
import sys

import apache_beam as beam
from apache_beam.metrics import Metrics
try:
    try:
        from apache_beam.options.pipeline_options import PipelineOptions
    except ImportError:
        from apache_beam.utils.pipeline_options import PipelineOptions
except ImportError:
    from apache_beam.utils.options import PipelineOptions
from PIL import Image
import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets import inception_v3 as inception
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io

import gcs_dataflow_utils
import numpy as np
import sys
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



class ExtractLabelIdsDoFn(beam.DoFn):
    """Extracts (uri, label_ids) tuples from CSV rows.
  """

    def start_bundle(self, context=None):
        self.label_to_id_map = {}

    # The try except is for compatiblity across multiple versions of the sdk
    def process(self, row, all_labels):
        try:
            row = row.element
        except AttributeError:
            pass
        if not self.label_to_id_map:
            for i, label in enumerate(all_labels):
                label = label.strip()
                if label:
                    self.label_to_id_map[label] = i

        # Row format is: image_uri(,label_ids)*
        if not row:
            skipped_empty_line.inc()
            return

        csv_rows_count.inc()
        uri = row[0]
        if not uri or not uri.startswith('gs://'):
            invalid_uri.inc()
            return

        # In a real-world system, you may want to provide a default id for labels
        # that were not in the dictionary.  In this sample, we simply skip it.
        # This code already supports multi-label problems if you want to use it.
        label_ids = []
        for label in row[1:]:
            try:
                label_ids.append(self.label_to_id_map[label.strip()])
            except KeyError:
                unknown_label.inc()

        labels_count.inc(len(label_ids))

        if not label_ids:
            unlabeled_image.inc()
        yield row[0], label_ids



class CustomLoadFromGCSasNumpyDoFn(beam.DoFn):

    def __init__(self,path,file_name_suffix, filesystem_depth):
        self.path = path
        self.file_name_suffix = file_name_suffix
        self.filesystem_depth = filesystem_depth

    def process(self, element):
        try:
            element = element.element
        except AttributeError:
            pass
        uri, label_ids = element
        path_splits = uri.split("/");
        imagename = '/'.join(path_splits[(-1+(-1)*self.filesystem_depth):])
        output_filename = os.path.join(self.path,imagename+self.file_name_suffix)
        with gcs_dataflow_utils.open_file_read_binary(output_filename) as f:
            try:
                embedding = np.load(f)
                yield uri, embedding, label_ids
            except IOError as error:
                print "Error on uri: "+output_filename+" relative to image "+uri
                print error
                raise Exception('Failed to use Google Cloud Storage GCS object: {}. The GCS object was relative to image: {}'.format(output_filename, uri))




class GenerateTFExamplesFromNumpyEmbeddingsDoFn(beam.DoFn):

    def process(self, element):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        try:
            element = element.element
        except AttributeError:
            pass

        uri, embedding, label_ids = element

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_uri': _bytes_feature([uri]),
            'embedding': _float_feature(embedding.ravel().tolist()),
        }))

        if label_ids:
            label_ids.sort()
            example.features.feature['label'].int64_list.value.extend(label_ids)

        yield example



def configure_pipeline(p, opt):
    """Specify PCollection and transformations in pipeline."""
    read_input_source = beam.io.ReadFromText(
      opt.input_path, strip_trailing_newlines=True)
    read_label_source = beam.io.ReadFromText(
      opt.input_dict, strip_trailing_newlines=True)
    labels = (p | 'Read dictionary' >> read_label_source)
    _ = (p
       | 'Read input' >> read_input_source
       | 'Parse input' >> beam.Map(lambda line: csv.reader([line]).next())
       | 'Extract label ids' >> beam.ParDo(ExtractLabelIdsDoFn(),beam.pvalue.AsIter(labels))
       | 'Load numpy from gcs' >> beam.ParDo(CustomLoadFromGCSasNumpyDoFn(opt.embeddings_path, '.npy',opt.filesystem_depth))
       | 'Generate TFExample' >> beam.ParDo(GenerateTFExamplesFromNumpyEmbeddingsDoFn())
       | 'Save to disk'>> beam.io.WriteToTFRecord(opt.output_path,
                                                  coder=beam.coders.ProtoCoder(tf.train.Example),
                                                  file_name_suffix='.tfrecord.gz'))


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
      '--input_dict',
      dest='input_dict',
      required=True,
      help='Input dictionary. Specified as text file uri. '
      'Each line of the file stores one label.')

    ### NEW STUFF ###
    parser.add_argument(
      '--embeddings_path',
      required=True,
      help='Input directory to take numpy embeddings from.')
    parser.add_argument(
     '--filesystem_depth',
     dest='filesystem_depth',
     required=True,
     type=int,
     help='Depth of the output filesystem wrt of the input filesystem.')
    ### END NEW STUFF ###

    parser.add_argument(
      '--output_path',
      required=True,
      help='Output directory to write results to.')
    parser.add_argument(
      '--project',
      type=str,
      help='The cloud project name to be used for running this pipeline')

    parser.add_argument(
      '--job_name',
      type=str,
      default='flowers-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
      help='A unique job identifier.')
    parser.add_argument(
      '--num_workers', default=20, type=int, help='The number of workers.')
    parser.add_argument('--cloud', default=False, action='store_true')
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
