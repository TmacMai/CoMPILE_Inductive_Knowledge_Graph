News: version 2 has been uploaded.


Our CoMPILE has two versions. 



#################################version 1#########################################

The first version is implemented based on GraIL (https://github.com/kkteru/grail), in which we evaluate our message passing model on the original inductive datasets proposed by the authors of the GraIL. We thank very much for their code sharing.

To run the code, firstly you need to unrar the data.rar and place the folder under CoMPILE_github.

To train the model (take FB15k-237 inductive v4 dataset as example):

     python train.py -d fb237_v4 -e compile_fb_v4_ind


To evaluate the AUC score of the trained model:

     python test_auc.py -d fb237_v4_ind -e compile_fb_v4_ind



To evaluate the Hits@10 score of the trained model:

     python test_ranking.py -d fb237_v4_ind -e compile_fb_v4_ind
     
     
#################################version 2#########################################

In version 2, we implement our inductive learning system, including the data filtering, directed subgraph extraction, and the message passing mechanism.

To be updated...
