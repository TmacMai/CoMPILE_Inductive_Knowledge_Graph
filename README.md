To train the model (take FB15k-237 inductive v4 dataset as example):

python train.py -d fb237_v4 -e compile_fb_v4_ind


To evaluate the AUC score of the trained model:

python test_auc.py -d fb237_v4_ind -e compile_fb_v4_ind



To evaluate the Hits@10 score of the trained model:

python test_ranking.py -d fb237_v4_ind -e compile_fb_v4_ind

