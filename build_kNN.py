# -*- coding: utf-8 -*-
import sys
import kNN_utilities as knn

def main(argv):
    train_file, test_file, k, sim_fn, out_file = argv
    train = open(train_file).readlines()
    test = open(test_file).readlines()
    k = int(k)
    sim_fn = int(sim_fn)
    
    # train and calculate distance
    ind, labels = knn.train_test_mat(train, test, sim_fn)
    train_ind = ind[:len(train)]
    test_ind = ind[len(train):]

    # evaluate on train set
    train_pred, train_conf = knn.evaluate(train, labels, train_ind, k)
    
    # evaluate on test set
    test_pred, test_conf = knn.evaluate(test, labels, test_ind, k)
    
    # generate sys_out file
    knn.gen_output(train_pred, test_pred, k, out_file)
    
    # print confusion matrix
    knn.print_conf_matrix(train_conf, test_conf)

if __name__=="__main__":
    main(sys.argv[1:])