
val_path="/home/leo/data/nactev/list/2018actev_val_file.txt"

#############################################
#--- training hyperparams ---
mainFolder="net_runs"
netType="ECO"
batch_size=5
num_segments=8
pretrained="both"
subFolder=${mainFolder}/${netType}_${num_segments}_${pretrained}/eco_lite_rgb_model_best.pth.tar
##################################################################### 
echo ${subFolder}



python3 -u model_class.py  ${val_path}  ${subFolder} ${num_segments} ${pretrained}




