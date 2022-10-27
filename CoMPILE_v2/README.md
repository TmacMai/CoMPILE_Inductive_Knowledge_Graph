

To generate new dataset and extract subgraph:

python3 ./train/inductive_subgraph8.py --prefix FB15k-237-v1  --data ./data/FB15k-237-inductive-v1/  --dataset FB15k-237-v1 --hop 3


To train the model:

python3 ./train/main_compile.py --prefix FB15k-237-v1 --data ./data/FB15k-237-inductive-v1  --dataset FB15k-237-v1 --hop 3 --train True --pretrained_emb False  --epochs_conv 20 --direct True --output_folder ./train/checkpoints/FB15k-237-v1/out/ --batch_size_conv 128 --test True


To test the model:

python3 ./train/main_compile.py --prefix FB15k-237-v1 --data ./data/FB15k-237-inductive-v1  --dataset FB15k-237-v1 --hop 3 --train False --pretrained_emb False  --epochs_conv 20 --direct True --output_folder ./train/checkpoints/FB15k-237-v1/out/ --batch_size_conv 128 --test True




















