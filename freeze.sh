freeze_graph \
    --input_graph=./eval.pb \
    --input_checkpoint=./local.ckpt \
    --output_graph=./frozen_eval.pb \
    --output_node_names=output
