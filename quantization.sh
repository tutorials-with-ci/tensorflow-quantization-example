tflite_convert \
    --output_file=./final.tflite \
    --graph_def_file=./frozen_eval.pb \
    --inference_type=QUANTIZED_UINT8 \
    --input_arrays=input \
    --output_arrays=prob \
    --mean_values=128 \
    --std_dev_values=127
