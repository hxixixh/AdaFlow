#!/bin/bash

{
# Check if the correct number of arguments are passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_demo_file_path> <output_path>"
    exit 1
fi

# Getting the input and output directories from the arguments
DEMO_FILE=$1
OUT_PATH=$2
LIBERO_PATH=$3

NUM_DEMOS=50

OUT_PATH_LIBERO="datasets/libero_rgb/libero_90/$OUT_PATH"
OUT_PATH_ROBOMIMIC_FORMAT="datasets/libero_robomimic_format_demo50/libero_90/$OUT_PATH"


for (( i=0; i<$NUM_DEMOS; i++))
do
    # Assigning GPU from 0 to 7 iteratively for each demo
    GPU=$((i % 8))
    
    export HYDRA_FULL_ERROR=1 
    export CUDA_VISIBLE_DEVICES=$GPU 
    export MUJOCO_EGL_DEVICE_ID=$GPU 
    python scripts/dataset_states_to_obs.py --use-actions --use-camera-obs --demo-file $DEMO_FILE --out_hdf5_path $OUT_PATH_LIBERO --demo_index $i --path_to_libero_lib $LIBERO_PATH &
done

wait  # This ensures that all background processes finish before the script exits

echo "All processes have completed."

# concatenate the demos
echo "Concatenating $OUT_PATH_LIBERO"
python scripts/concat_demos.py --demo-file $DEMO_FILE --out_hdf5_path $OUT_PATH_LIBERO --num_demos=$NUM_DEMOS --path_to_libero_lib $LIBERO_PATH

echo "Data concatenation process has completed."
wait

# convert libero data to robomimic format
python scripts/convert_libero_to_robomimic_format.py --input_file=$OUT_PATH_LIBERO --output_file=$OUT_PATH_ROBOMIMIC_FORMAT

# convert to abs action
python scripts/convert_libero_to_abs_action.py --input=$OUT_PATH_ROBOMIMIC_FORMAT --path_to_libero_lib=$LIBERO_PATH --output="${OUT_PATH_ROBOMIMIC_FORMAT%.hdf5}_abs.hdf5" 


python scripts/convert_hdf5_to_jpg.py --dataset-folder="datasets/libero_robomimic_format_demo50/libero_90" --save_rgb --save_lowdim 

echo "JPG conversion processes has completed."

exit
}