#!/bin/bash

GCS_PATH="gs://alex-s2t-test/adversarial/datasets/six_labels"
LABEL_COUNT=6

cd convolutional-multilevel-training

python -u dataflow-generate_labelled_tfrecords_from_numpy.py \
    --input_dict "$GCS_PATH/labels/labels.txt" \
    --input_path "$GCS_PATH/labels/train_set.csv" \
    --embeddings_path "$GCS_PATH/inception_resnet/preproc/numpy/PreLogitsFlatten/" \
    --filesystem_depth=1 \
    --output_path "$GCS_PATH/inception_resnet/preproc/tfrecord/train" \
    --cloud \
    --worker_machine_type n1-highcpu-8 \
    --num_workers 30 \
    --job_name="generate-tfrecords-train" \
  --setup_file ./setup.py && \
python -u dataflow-generate_labelled_tfrecords_from_numpy.py \
    --input_dict "$GCS_PATH/labels/labels.txt" \
    --input_path "$GCS_PATH/labels/eval_set.csv" \
    --embeddings_path "$GCS_PATH/inception_resnet/preproc/numpy/PreLogitsFlatten/" \
    --filesystem_depth=1 \
    --output_path "$GCS_PATH/inception_resnet/preproc/tfrecord/eval" \
    --cloud \
    --worker_machine_type n1-highcpu-8 \
    --num_workers 30 \
    --job_name="generate-tfrecords-eval" \
    --setup_file ./setup.py
gcloud beta ml-engine jobs submit training "transferlearning`date +%s`" \
    --stream-logs \
    --module-name trainer.task \
    --package-path trainer \
    --staging-bucket "gs://mirco-dataflow-stage" \
    --region us-central1 \
    --runtime-version=1.0 \
    -- \
    --label_count $LABEL_COUNT \
    --output_path "$GCS_PATH/inception_resnet/training" \
    --trained_checkpoint_path "gs://alex-s2t-test/adversarial/nets_ckpt/inception_resnet_v2_2016_08_30.ckpt" \
    --eval_data_paths "$GCS_PATH/inception_resnet/preproc/tfrecord/eval*" \
    --train_data_paths "$GCS_PATH/inception_resnet/preproc/tfrecord/train*" && \
python tester/test_saved_model_with_tensorflow_serving.py \
    --model_path  "$GCS_PATH/inception_resnet/training/model" \
    --labels_file "$GCS_PATH/labels/labels.txt" \
    --images_file "$GCS_PATH/labels/eval_set.csv"  \
    --output_path "$GCS_PATH/inception_resnet/statistics/" \
    --final_layer_type Softmax \
    --no_na True \
    --write_predictions True
    
    
cd ..