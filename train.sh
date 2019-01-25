

# Parameters!
mainFolder="net_runs"
subFolder="ECO_lite_run1"
snap_pref="eco_lite"

train_path="/home/leo/data/nactev/list/2018actev_train_file.txt"
val_path="/home/leo/data/nactev/list/2018actev_val_file.txt"

#############################################
#--- training hyperparams ---
dataset_name="actev"
netType="ECO"
batch_size=5
learning_rate=0.001
num_segments=8
dropout=0.3
iter_size=4
num_workers=5
pretrained="both"
subFolder=${netType}_${num_segments}_${pretrained}
##################################################################### 
mkdir -p ${mainFolder}/${subFolder}

echo "Current network folder: "
echo ${mainFolder}/${subFolder}



python3 -u main.py ${dataset_name} RGB ${train_path} ${val_path}  --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --lr_steps 15 30 --epochs 240 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder}/${snap_pref} --consensus_type identity --eval-freq 1   --pretrained_parts ${pretrained} --no_partialbn --nesterov "True" --resume ${mainFolder}/${subFolder} 2>&1 | tee -a ${mainFolder}/${subFolder}/log.txt    




