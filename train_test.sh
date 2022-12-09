#!/usr/bin/env bash

# License:
#     MIT License
#
#     Copyright (c) 2022 HUAWEI CLOUD

# train_test.sh

echo "Initializing the network"
python -m c2far.ml.init_lstm --ncoarse_bins 20 --coarse_low -0.1 --coarse_high 1.3 --nfine_bins 20 --nfine2_bins 20 --nhidden 128 --nlayers 2 --extremas --o resources/results/init_models/init_lstm.128.20.20.20.x.net

echo "Training the network"
python -m c2far.ml.train_lstm --train_vcpus azure_demand/azure.vcpus.1hour.dedupe --train_mem azure_demand/azure.memory.1hour.dedupe --train_offs azure_demand/azure.1hour~168~168~24.train --train_loss_end_pt_s 1728000 --test_loss_start_pt_s 1728000 --test_loss_end_pt_s 1987200 --test_vcpus azure_demand/azure.vcpus.1hour.dedupe --test_mem azure_demand/azure.memory.1hour.dedupe --test_offs azure_demand/azure.1hour~168~168~24.dev --ncoarse_bins 20  --coarse_low -0.1 --coarse_high 1.3 --nfine_bins 20 --nfine2_bins 20 --extremas --device cuda:1 --lstm_model0 resources/results/init_models/init_lstm.128.20.20.20.x.net --lr 2e-3 --weight_decay 1e-06 --out_dir resources/results --ntrain_checkpoint 32768 --train_batch_size 64 --test_batch_size 32 --cache_testset --csize 168 --nsize 168 --nstride 24 --gsize 24 --lstm_dropout 1e-3 --run_gen_eval --nsamples 25 --confidence 1 --test_eval_period 1 --model_save_period 1 --ntest_checkpoint 32768 --max_num_iters 12 --gen_bdecoder uniform --seed 0

echo "Evaluating the results"
# Should get MAE < 100 (ND < 2.9%), which is already better than baselines and flat binning:
python -m c2far.ml.lstm_evaluation --test_loss_start_pt_s 1987200 --test_loss_end_pt_s 2246400 --test_vcpus azure_demand/azure.vcpus.1hour.dedupe --test_mem azure_demand/azure.memory.1hour.dedupe --test_offs azure_demand/azure.1hour~168~168~24.test --ncoarse_bins 20  --coarse_low -0.1 --coarse_high 1.3 --nfine_bins 20 --nfine2_bins 20 --extremas --device cuda:1 --lstm_model resources/results/_TripleCE_20_20_20_extremas_True_/resources_results_init_models_init_lstm_128_20_20_20_x_net/azure.1hour~168~168~24.train/*/model.*.pt --logging_dir resources/results/evaluation --test_batch_size 32 --csize 168 --nsize 168 --nstride 24 --gsize 24 --run_gen_eval --nsamples 500 --confidence 80 --ntest_checkpoint -1 --gen_bdecoder uniform --ntest_checkpoint -1 --seed 0

echo "Evaluating the tuned model"
# Should get roughly the same number as in the paper:
python -m c2far.ml.lstm_evaluation --test_loss_start_pt_s 1987200 --test_loss_end_pt_s 2246400 --test_vcpus azure_demand/azure.vcpus.1hour.dedupe --test_mem azure_demand/azure.memory.1hour.dedupe --test_offs azure_demand/azure.1hour~168~168~24.test --ncoarse_bins 16  --coarse_low -0.1 --coarse_high 1.3 --nfine_bins 11 --nfine2_bins 94 --extremas --device cuda:0 --lstm_model resources/models/model.trained.nhidden=130.bins=16.11.94.x.pt --logging_dir resources/results/evaluation --test_batch_size 32 --csize 168 --nsize 168 --nstride 24 --gsize 24 --run_gen_eval --nsamples 500 --confidence -1 --ntest_checkpoint -1 --gen_bdecoder uniform --ntest_checkpoint -1 --seed 0
