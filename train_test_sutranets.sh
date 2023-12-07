#!/usr/bin/env bash

# License:
#     MIT License
#
#     Copyright (c) 2023 HUAWEI CLOUD

# train_test_sutranets.sh

DEVICE="cuda:1"
# Use 750 in paper, with early stopping:
MAX_NUM_ITERS=1
# Use 8192 in paper:
NDEV_CHECKPOINT=8
# Use 2208000 in paper (53.5% of total):
NTEST_CHECKPOINT=500
# Use 500 in paper:
NTEST_SAMPLES=50

if true; then
	# Make the files from the parts, if not already made:
	for basen in azure.vcpus.5min.dedupe azure.memory.5min.dedupe azure.5min.2016~288.train azure.5min.2016~288.dev azure.5min.2016~288.test; do
		if [ -f azure_demand/$basen ]; then
			echo "$basen ready"
		else
			echo "Making $basen"
			cat azure_demand/file_parts/$basen.* > azure_demand/$basen
		fi
	done
fi

INIT_MODEL=resources/results/init_models/triple_lstm.12.12.12.64.1.nsub_series.6.backfill.alt.net
if true; then
	if [ -d $INIT_MODEL ]; then
		rm -r $INIT_MODEL
	fi
	python -m sutranets.ml.init_lstm --ncoarse_bins 12 --nfine_bins 12 --nfine2_bins 12 --extremas --coarse_low -0.08 --coarse_high 1.2 --nhidden 64 --nlayers 1 --nsub_series 6 --sub_csize 336 --mv_backfill --out_fn $INIT_MODEL
fi

if true; then
	echo ">>> Running training"
	if [ -d resources/results/values ]; then
		rm -r resources/results/values
	fi
	python -m sutranets.ml.train_lstm --train_trace_paths azure_demand/azure.vcpus.5min.dedupe azure_demand/azure.memory.5min.dedupe --train_offs azure_demand/azure.5min.2016~288.train --train_loss_end_pt_s 1728000 --test_loss_start_pt_s 1728000 --test_loss_end_pt_s 1987200 --test_trace_paths azure_demand/azure.vcpus.5min.dedupe azure_demand/azure.memory.5min.dedupe --test_offs azure_demand/azure.5min.2016~288.dev --device $DEVICE --train_batch_size 128 --test_batch_size 32 --ntrain_checkpoint 8192 --ntest_checkpoint $NDEV_CHECKPOINT --csize 2040 --nsize 2016 --nstride 288 --gsize 288 --ncoarse_bins 12 --nfine_bins 12 --nfine2_bins 12 --coarse_low -0.08 --coarse_high 1.2 --lr 1e-2 --weight_decay 1e-4 --out_dir resources/results --lstm_model0 $INIT_MODEL --gen_bdecoder uniform --test_warmup_period 0 --test_eval_period 1 --model_save_period 1 --plot_period 1 --extremas --nsub_series 6 --sub_csize 336 --seed 1 --sample_period_s 300 --num_workers 16 --mv_backfill --max_num_iters $MAX_NUM_ITERS --cache_testset --run_gen_eval --dont_run_1step_eval --nsamples 25 --confidence_pct 80
fi

if true; then
	echo ">>> Running evaluation"
	# Use the model trained in the previous step:
	model="resources/results/values/_MultivLoss__TripleCE_12_12_12_extremas_True__/resources_results_init_models_triple_lstm_12_12_12_64_1_nsub_series_6_backfill_alt_net/azure.5min.2016~288.train/*/model.*.pt"
	python -m sutranets.ml.lstm_evaluation --train_trace_paths azure_demand/azure.vcpus.5min.dedupe azure_demand/azure.memory.5min.dedupe --train_offs azure_demand/azure.5min.2016~288.train --train_loss_end_pt_s 1728000 --test_loss_start_pt_s 1987200 --test_loss_end_pt_s 2246400 --test_trace_paths azure_demand/azure.vcpus.5min.dedupe azure_demand/azure.vcpus.5min.dedupe --test_offs azure_demand/azure.5min.2016~288.test --device $DEVICE --train_batch_size 128 --test_batch_size 10 --csize 2040 --nsize 2016 --nstride 288 --gsize 288 --run_gen_eval --confidence 80 --nsamples $NTEST_SAMPLES --sample_period_s 300 --gen_bdecoder uniform --ntest_checkpoint $NTEST_CHECKPOINT --ncoarse_bins 12 --nfine_bins 12 --nfine2_bins 12 --coarse_low -0.08 --coarse_high 1.2 --dont_run_1step --num_workers 16 --seed 0 --extremas --lstm_model $model --nsub_series 6 --sub_csize 336 --mv_backfill
fi

if true; then
	echo ">>> Cleaning up"
	rm azure_demand/azure.5min.2016~288.dev
	rm azure_demand/azure.5min.2016~288.test
	rm azure_demand/azure.5min.2016~288.train
	rm azure_demand/azure.memory.5min.dedupe
	rm azure_demand/azure.vcpus.5min.dedupe
	rm -r resources/results/values
fi
