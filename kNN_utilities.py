# -*- coding: utf-8 -*-
from collections import Counter
import re
import numpy as np
import scipy.spatial.distance as dist

# for processing data format
re_label = re.compile(r'^(\S+)\s')
re_feat = re.compile(r'(\S+):\d+')
re_feat_num = re.compile(r'(\S+):(\d+)')
    
    
def make_vec(line, vocab):
    """line is a document from text file in svmlight format
    vocab is an ordered list (produced by train_test_mat)
    returns vector of occurences of each word (preserving vocab order)
    """
    feats = {g[0]:int(g[1]) for g in re_feat_num.findall(line)}
    
    vec = np.zeros(len(vocab), dtype=int)
    for i in range(len(vocab)):
        if vocab[i] in feats:
            vec[i] = feats[vocab[i]]
            
    return(vec)

def train_test_mat(train, test, dist_type):
    """train and test are lists of lines
    returns ordered indices and corresponding labels
    """
    # extract vocab and class labels
    vocab = set()
    labels = []
    train_test = []
    for line in train:
        feats = re_feat.findall(line)
        labels.append(re_label.search(line).group(1))
        
        for feat in feats: vocab.add(feat)
      
    vocab = sorted(list(vocab))
    # create train vectors
    for line in train:
        train_test.append(make_vec(line, vocab))
        
    # create test vectors
    for line in test:
        train_test.append(make_vec(line, vocab))
        
    # determine distance metric
    if dist_type == 1:
        met = "euclidean"
        
    elif dist_type == 2:
        met = "cosine"
        
    else: return("ERROR: distance_type must be 1 (Euclidean) or 2 (cosine)")
    
    # generate distance matrix
    dist_mat = dist.squareform(dist.pdist(train_test, metric=met))
    
    # cut out columns corresponding to test observations
    dist_mat = np.delete(dist_mat, slice(len(train),dist_mat.shape[1]), axis=1)
    
    ord_ind = np.argsort(dist_mat, axis=1)

    return(ord_ind, labels)

def generate_results(train_labels, ord_ind, k):
    results = []
    
    for line in ord_ind:
        indices = line[:k]
        labels = [train_labels[i] for i in indices]
        results.append(labels)
        
    return(results)

def evaluate(data, labels, ord_ind, k):
    """data is lines from text file (labelled)
    labels is from training set (list generated by train_test_mat)
    returns list of [true label, [predictions]] pairs and confusion matrix
    """
    all_labels = set(labels)
    conf_matrix = {lab1:{lab2:0 for lab2 in all_labels} for lab1 in all_labels}
    k_labels = generate_results(labels, ord_ind, k)
    results = []
    
    for i in range(len(data)):
        t_label = re_label.search(data[i]).group(1)
        p_labels = Counter(k_labels[i])
        ord_p = p_labels.most_common()
        
        results.append([t_label, ord_p])
        conf_matrix[t_label][ord_p[0][0]] += 1
        
    return(results, conf_matrix)
   

def gen_output(train_results, test_results, k, file):
    """train_results and test_results are lists produced by evaluate
    writes to file
    """
    f_out = open(file, mode="w+")     
    
    f_out.write("%%%%% training data:\n")
    i = 0
    for r in train_results:
        f_out.write("array:{} {}".format(i, r[0]))
        
        for pair in r[1]:
            f_out.write(" {} {}".format(pair[0], pair[1]/k))
    
        f_out.write("\n")
        i += 1
        
    f_out.write("%%%%% test data:\n")
    i = 0
    for r in test_results:
        f_out.write("array:{} {}".format(i, r[0]))
        
        for pair in r[1]:
            f_out.write(" {} {}".format(pair[0], pair[1]/k))
    
        f_out.write("\n")
        i += 1
    
    f_out.close()
    
def print_conf_matrix(train_conf, test_conf):
    """train_conf ans test_conf are confusion matrices (dictionaries)
    generated by evaluate
    PRINTS to stdout
    """
    labels = sorted(list(train_conf.keys()))
    
    print("Confusion matrix for the training data:\nrow is the truth, column is the system output\n")
    print("\t\t"+" ".join(labels))
    
    for label in labels:
        curr = train_conf[label]
        print(label +"\t" +"\t".join([str(curr[i]) for i in labels]))
        
    total_train = sum([sum(list(train_conf[i].values())) for i in labels])
    train_correct = sum([train_conf[i][i] for i in labels])
    
    print("\nTraining accuracy={}\n\n".format(train_correct/total_train))
    
    print("Confusion matrix for the test data:\nrow is the truth, column is the system output\n")
    print("\t\t"+" ".join(labels))
    
    for label in labels:
        curr = test_conf[label]
        print(label +"\t" +"\t".join([str(curr[i]) for i in labels]))
        
    total_test = sum([sum(list(test_conf[i].values())) for i in labels])
    test_correct = sum([test_conf[i][i] for i in labels])
    
    print("\nTest accuracy={}".format(test_correct/total_test))