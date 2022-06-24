#########################################
This folder contains details for each training version.
Data path:

------------------------------------------


Directory description:
-test_on_human_data
This folder contains analysis on human data.
See README in it for detail.

-temporary_testing_version
This folder contains temporary testing files.
The codes root from Edwinn's codes with step-by-step modification.
See README in it for detail.

#########################################
Future training session (by YS)

(1) Train the model with variable sequence lengths:
https://danijar.com/variable-sequence-lengths-in-tensorflow/
(2) Don't use dropout for preference inference.
(3) Reconsider the data format of 'data_preference_predictions'. Maybe I should try more combination?
(4)
########################################
Finished training session (v1,commit 603c34)
Time: 2022/06/22
Author: Elaine
Output file name: /cache_dS001_v1_commit_603c34
Info: train_9000_test_preference_predictor_with_another_1000


train: proportion_accuracy()
Matches: 2662/7200
Accuracy: 36.97%
56 vali batches in total...
Model restored from  /home/.bml/Data/Bank1/ToMNET/tomnet-project/tomnet2/tomnet2/models/working_model/test_on_simulation_data/training_result/caches/cache_dS001_v1_commit_603c34/train/model.ckpt-8999
0 batches finished!
10 batches finished!
20 batches finished!
30 batches finished!
40 batches finished!
50 batches finished!

vali: proportion_accuracy()
Matches: 157/896
Accuracy: 17.52%
56 test batches in total...
Model restored from  /home/.bml/Data/Bank1/ToMNET/tomnet-project/tomnet2/tomnet2/models/working_model/test_on_simulation_data/training_result/caches/cache_dS001_v1_commit_603c34/train/model.ckpt-8999
0 batches finished!
10 batches finished!
20 batches finished!
30 batches finished!
40 batches finished!
50 batches finished!

test: proportion_accuracy()
Matches: 149/896
Accuracy: 16.63%
------------------------------------



########################################
Finished training session (v24, commit 014d79)[At working_model]
Time: 2020/01/15
Author: Elaine
Output file name: /cache_S003b_v24_commit_014d79
Info: file10000_tuning_batch16_train_step_10K_INIT_LR_10-4

(1) Follow v23 but train it with S003b data.
Note:
(1) The result is similar to v23.

(2)
accurary	mode
97.09%	train_proportion
75.91%	vali_proportion
75.3%	test_proportion


(3)
Traj_S003b_Query_S003b_subset96
avg_prediction_probability	ground_truth_label_count	prediction_count	accuracy_data_set
0.21	19	19	85.42
0.25	21	25	85.42
0.14	19	12	85.42
0.41	37	40	85.42

Traj_S003b_Query_Stest_subset96:
prediction_proportion	avg_prediction_probability	ground_truth_label_count	prediction_count
0.25	0.2	0	24
0.15	0.2	0	14
0	0.05	0	0
0.6	0.55	0	58
########################################

