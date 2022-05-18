import pandas as pd
import numpy as np
import tensorflow as tf
import time
import argparse
from EGES_model import EGES_Model
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_sampled", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--root_path", type=str, default='./data_cache/')
    parser.add_argument("--num_feat", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--outputEmbedFile", type=str, default='./embedding/EGES.embed')
    args = parser.parse_args()

    # read train_data
    print('read features...')
    start_time = time.time()

    # len 34048 side_info
    # S = 4
    # [
    #   [  0,  24, 314,  67],
    #   ...,
    #   [34047,   418,  3854,    18]
    # ]
    side_info = np.loadtxt(args.root_path + 'sku_side_info.csv', dtype=np.int32, delimiter='\t')
    # len 12999900
    # [
    #   [15104, 12212],
    #   ...,
    #   []
    # ]
    all_pairs = np.loadtxt(args.root_path + 'all_pairs', dtype=np.int32, delimiter=' ')
    # [34048, 3663, 4786, 80] 每个 side_info 特征的数目
    feature_lens = []
    for i in range(side_info.shape[1]):
        tmp_len = len(set(side_info[:, i]))
        feature_lens.append(tmp_len)
    end_time = time.time()
    print('time consumed for read features: %.2f' % (end_time - start_time))

    #
    EGES = EGES_Model(
        # 样本数
        len(side_info),
        # S
        args.num_feat,
        # 特征数
        feature_lens,
        # 窗口大小
        n_sampled=args.n_sampled,
        # d embedding向量维度
        embedding_dim=args.embedding_dim,
        lr=args.lr
    )

    # init model
    print('init...')
    start_time = time.time()
    init = tf.global_variables_initializer()
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    sess = tf.Session(config=config_tf)
    sess.run(init)
    end_time = time.time()
    print('time consumed for init: %.2f' % (end_time - start_time))

    print_every_k_iterations = 100
    loss = 0
    iteration = 0
    start = time.time()

    #
    # 6347 对所有的 pair 除以 batch_size
    max_iter = len(all_pairs) // args.batch_size * args.epochs
    #
    for iter in range(max_iter):
        #
        iteration += 1
        #
        # [[19562], ..., []]
        batch_features, batch_labels = next(graph_context_batch_iter(all_pairs, args.batch_size, side_info,
                                                                     args.num_feat))
        #
        feed_dict = {input_col: batch_features[:, i] for i, input_col in enumerate(EGES.inputs[:-1])}
        #
        feed_dict[EGES.inputs[-1]] = batch_labels
        #
        _, train_loss = sess.run([EGES.train_op, EGES.cost], feed_dict=feed_dict)

        #
        loss += train_loss

        #
        if iteration % print_every_k_iterations == 0:
            #
            end = time.time()
            #
            e = iteration * args.batch_size // len(all_pairs)
            #
            print("Epoch {}/{}".format(e, args.epochs),
                  "Iteration: {}".format(iteration),
                  "Avg. Training loss: {:.4f}".format(loss / print_every_k_iterations),
                  "{:.4f} sec/batch".format((end - start) / print_every_k_iterations))
            #
            loss = 0
            start = time.time()

    print('optimization finished...')
    saver = tf.train.Saver()
    saver.save(sess, "checkpoints/EGES")

    #
    feed_dict_test = {input_col: list(side_info[:, i]) for i, input_col in enumerate(EGES.inputs[:-1])}
    #
    feed_dict_test[EGES.inputs[-1]] = np.zeros((len(side_info), 1), dtype=np.int32)
    #
    embedding_result = sess.run(EGES.merge_emb, feed_dict=feed_dict_test)
    #
    print('saving embedding result...')
    #
    write_embedding(embedding_result, args.outputEmbedFile)

    print('visualization...')
    plot_embeddings(embedding_result[:5000, :], side_info[:5000, :])
