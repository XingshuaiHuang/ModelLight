#######################################################ModelLight#######################################################
## **************** meta_train ********************
memo="mbmrl_train"
model_type="mbmrl"
traffic_group="train_all"
python meta_train.py --memo ${memo} --algorithm MBMRL --pre_train_model_name ${model_type} \
--update_start 0 --test_period 1 --traffic_group ${traffic_group} --epochs 200 --num_img_trans 36 --num_img_round 10 \
--model_sample_size 30 --meta_update_period 10 --run_round 10
echo "============================= mbmrl: meta_train complete ============================="

### **************** meta_test ********************
#memo="mbmrl_test"
#model_type="mbmrl" # metalight or maml or random or mbmrl
#traffic_group="valid_all" # valid_all (task1) or city_all (task2: manually change homo or hete)
#python meta_test.py --memo ${memo} --algorithm FRAPPlus --pre_train --pre_train_model_name ${model_type} --run_round 1 \
#--num_process 2 --update_start 0 --test_period 1 --traffic_group ${traffic_group}
#echo "============================= meta_test complete ============================="
#python summary.py --memo ${memo} --type meta_test
#echo "============================= summary complete ============================="