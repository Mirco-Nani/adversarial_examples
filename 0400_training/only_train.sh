#!/bin/bash

GCS_PATH="gs://alex-s2t-test/adversarial/datasets/six_labels"
LABEL_COUNT=6

cd convolutional-multilevel-training

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
    --train_data_paths "$GCS_PATH/inception_resnet/preproc/tfrecord/train*"
    
cd ..

