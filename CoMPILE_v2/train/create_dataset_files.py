import numpy as np
import pandas as pd
import argparse

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("--data", help="data directory")
    args = args.parse_args()
    return args

args = parse_args()

def getID():
    folder = './data/%s/attentionEmb/'%args.data
    lstEnts = {}
    lstRels = {}
    with open(folder + 'train_inductive.txt') as f, open(folder + 'train_marked_inductive.txt', 'w') as f2:
        count = 0
        for line in f:
            line = line.strip().split('\t')
            line = [i.strip() for i in line]
            # print(line[0], line[1], line[2])
            if line[0] not in lstEnts:
                lstEnts[line[0]] = len(lstEnts)
            if line[1] not in lstRels:
                lstRels[line[1]] = len(lstRels)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            count += 1
            f2.write(str(line[0]) + '\t' + str(line[1]) +
                     '\t' + str(line[2]) + '\n')
        print("Size of train_marked set set ", count)

    with open(folder + 'valid_inductive.txt') as f, open(folder + 'valid_marked_inductive.txt', 'w') as f2:
        count = 0
        for line in f:
            line = line.strip().split('\t')
            line = [i.strip() for i in line]
            # print(line[0], line[1], line[2])
            if line[0] not in lstEnts:
                lstEnts[line[0]] = len(lstEnts)
            if line[1] not in lstRels:
                lstRels[line[1]] = len(lstRels)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            count += 1
            f2.write(str(line[0]) + '\t' + str(line[1]) +
                     '\t' + str(line[2]) + '\n')
        print("Size of valid_marked set set ", count)

    with open(folder + 'test_inductive.txt') as f, open(folder + 'test_marked_inductive.txt', 'w') as f2:
        count = 0
        for line in f:
            line = line.strip().split('\t')
            line = [i.strip() for i in line]
            # print(line[0], line[1], line[2])
            if line[0] not in lstEnts:
                lstEnts[line[0]] = len(lstEnts)
            if line[1] not in lstRels:
                lstRels[line[1]] = len(lstRels)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            count += 1
            f2.write(str(line[0]) + '\t' + str(line[1]) +
                     '\t' + str(line[2]) + '\n')
        print("Size of test_marked set set ", count)

    wri = open(folder + 'entity2id_inductive.txt', 'w')
    for entity in lstEnts:
        wri.write(entity + '\t' + str(lstEnts[entity]))
        wri.write('\n')
    wri.close()

    wri = open(folder + 'relation2id_inductive.txt', 'w')
    for entity in lstRels:
        wri.write(entity + '\t' + str(lstRels[entity]))
        wri.write('\n')
    wri.close()

    entity_df = pd.read_csv(folder+'entity2id_inductive.txt',sep='\t',header=None,names=['entity','index'])
    feature_df = pd.read_csv('./data/%s/feature.csv'%args.data)
    feature_df = feature_df.drop_duplicates(['entity']).reset_index(drop=True)
    for col in feature_df.columns:
        if col not in ['entity','index','type']:
            feature_df[col] = (feature_df[col]-feature_df[col].mean()) / feature_df[col].std()
    feature_df = pd.concat([feature_df.drop('type',axis=1),pd.get_dummies(feature_df['type'])],axis=1)
    entity_df = entity_df.merge(feature_df,how='left',on='entity')
    entity_df[[col for col in entity_df.columns if col not in ['entity','index','type']]].to_csv(folder+'entity2vec_inductive.txt',sep='\t',header=None,index=False)
    relation_df = pd.read_csv(folder+'relation2id_inductive.txt',sep='\t',header=None,names=['entity','index'])
    #relation_embeddings = np.random.randn(len(relation_df), 50)
    relation_embeddings = np.concatenate([np.random.randn(len(relation_df), 50),np.eye(len(relation_df))],axis=1)
    np.savetxt(folder+'relation2vec_inductive.txt', relation_embeddings,fmt='%f',delimiter='\t')

getID()
