import numpy as np
import tensorflow as tf


class EGES_Model:
    def __init__(self, num_nodes, num_feat, feature_lens, n_sampled=100, embedding_dim=128, lr=0.001):
        """
        :param num_nodes:  V
        :param num_feat:  S
        :param feature_lens:
        :param n_sampled: k
        :param embedding_dim: d
        :param lr:
        """
        # k=100 窗口大小
        self.n_samped = n_sampled
        # S=4
        self.num_feat = num_feat
        # 这里是每个 side_info 的特征数 [34048, 3663, 4786, 80]
        self.feature_lens = feature_lens
        # d=128
        self.embedding_dim = embedding_dim
        # V=34048
        self.num_nodes = num_nodes
        # 0.001
        self.lr = lr
        # shape=(34048, 128) todo 大小是 [V, d]
        # 这里相当于对于每个 item_id 都做了一次 embedding
        # 这是第一个表，传入 item_id，查询出长度为 embedding_dim 的向量
        self.softmax_w = tf.Variable(tf.truncated_normal((num_nodes, embedding_dim), stddev=0.1), name='softmax_w')
        # shape=(34048,) todo 大小是 [V,]
        # 这是 embedding 过程中的 bias
        # 第二个表，传入 item_id 查询出长度为 1 的向量
        self.softmax_b = tf.Variable(tf.zeros(num_nodes), name='softmax_b')
        #
        # 输入的初始化，4 个 side_info feature 和 1 个 label
        self.inputs = self.input_init()
        #
        # [B, S+1]
        # embedding 的初始化也简单，就是把每个独立的特征都建一个表，组成一个 list
        # 每一个表都是 <feature_num, d>
        self.embedding = self.embedding_init()
        #
        # A [V, S] 长度为 node_num, 宽度为边数
        # shape=(34048, 4) 给每个 item 特征都赋予一个权重，A_{ij} 表示第 i 个 item 的第 j 类 side information 的权重
        # 这里相当于拿出来一个节点，边数为 S，给这个长度为 S 的列表每个元素赋予一个权重
        # 第三个表，传入 item_id ，查出长度为 num_feat 的向量
        self.alpha_embedding = tf.Variable(tf.random_uniform((num_nodes, num_feat), -1, 1))
        #
        # 对传入的 side_info 信息，查出 embedding 向量，然后根据 self.alpha_embedding 进行加权，最终得到 shape=(?, 128)
        self.merge_emb = self.attention_merge()
        self.cost = self.make_skipgram_loss()
        # self.train_op = tf.train.AdagradOptimizer(lr).minimize(self.cost)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)

    def embedding_init(self):
        cat_embedding_vars = []
        for i in range(self.num_feat):
            # [shape=(34048, 128), shape=(3663, 128), shape=(4786, 128), shape=(80, 128)]
            # 这里的 embedding_var 就相当于我们的 table
            # 第四个表，分为 num_feat 个小表，传入该 feature 的数目，查出 embedding_dim 长度的向量
            embedding_var = tf.Variable(tf.random_uniform((self.feature_lens[i], self.embedding_dim), -1, 1),
                                        name='embedding' + str(i),
                                        trainable=True)
            cat_embedding_vars.append(embedding_var)
        return cat_embedding_vars

    def attention_merge(self):
        # [shape=(?, 128), shape=(?, 128), shape=(?, 128), shape=(?, 128)]
        embed_list = []
        for i in range(self.num_feat):
            # 输入一条样本，一共四个特征，每个特征都去 embedding 的列表里面查一下对于的 embedding，当然查出来的都是 128 维的，shape=(?, 128)
            cat_embed = tf.nn.embedding_lookup(self.embedding[i], self.inputs[i])
            embed_list.append(cat_embed)
        # 这就是查出来的 embedding shape=(?, 128, 4)
        stack_embed = tf.stack(embed_list, axis=-1)
        # attention merge
        # 用 item_id 去查条样本各个 side_info 的权重，shape=(?, 4) 这个 4 就是 S，即边数
        alpha_embed = tf.nn.embedding_lookup(self.alpha_embedding, self.inputs[0])
        # shape=(?, 1, 4) 相乘之前必须扩展权重参数的维度
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        # 对各个 side_info 的权重先做 exp 再求和，得到 shape=(?, 1)这就是一个常数
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        # shape=(?, 128)
        merge_emb = tf.reduce_sum(
            # (?, 128, 4)
            # (?, 1, 4)
            # 对于每行先乘上 side_info 的 exp 权重，得到 (?, 128, 4) 这就是加权后的一个 item 的 S=4 个权重向量，下一步就是如何把这 S=4 个权重向量合并成一个
            stack_embed * tf.exp(alpha_embed_expand),
            # 在最后一个维度上 reduce_sum，得到 (?, 128)
            axis=-1
        ) / alpha_i_sum
        # 最终得到 (?, 128) 这就是组合好的向量
        return merge_emb

    def input_init(self):
        input_list = []
        for i in range(self.num_feat):
            input_col = tf.placeholder(tf.int32, [None], name='inputs_' + str(i))
            input_list.append(input_col)
        input_list.append(tf.placeholder(tf.int32, shape=[None, 1], name='label'))
        # [shape=(?,), shape=(?,), shape=(?,), shape=(?,), label=shape=(?,),]
        return input_list

    def make_skipgram_loss(self):
        loss = tf.reduce_mean(
            # https://blog.csdn.net/qq_41978896/article/details/109164450
            tf.nn.sampled_softmax_loss(
                # (V, d) 这就是 w，即要训练的参数
                weights=self.softmax_w,
                # (V,) 这是训练的剩余产物
                biases=self.softmax_b,
                # (?, 1) todo 这个 label 是什么？
                # 标签，简单理解为正样本的id
                labels=self.inputs[-1],
                # (?, 128) todo 这是输入的那个集合的 embedding
                # 输入，简单理解为一个batch中用户的表征
                inputs=self.merge_emb,
                # k=100 就是窗口宽度，负样本数
                num_sampled=self.n_samped,
                # V 这是整张词汇表
                # 定义有多少类别，方便负样本从不同的类别中采样，每个 item 就是一个类别，所以有 V 个类别
                num_classes=self.num_nodes,
                # 在负采样 softmax 中的实际分类（正负），默认为 1 即二分类
                # todo 这个意义没搞清
                # 表示单标签
                num_true=1,
                # 这是采样的办法
                # 候选采样
                # 采样负样本时候的概率
                sampled_values=tf.random.uniform_candidate_sampler(
                    # 正样本的 id
                    true_classes=tf.cast(self.inputs[-1], tf.int64),
                    # 实际的正样本个数
                    # 单标签
                    num_true=1,
                    # 窗口宽度 k
                    # 采样的数目
                    num_sampled=self.n_samped,
                    # 采样的结果无重复 id
                    unique=True,
                    # 取词的范围
                    range_max=self.num_nodes
                )
            ))

        return loss
