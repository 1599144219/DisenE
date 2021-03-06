import torch
import numpy as np
import time


class Corpus:
    def __init__(self, args, train_data, validation_data, test_data, link_data, entity2id, relation2id,
                 batch_size, valid_to_invalid_samples_ratio):
        self.train_triples = train_data
        self.validation_triples = validation_data
        self.test_triples = test_data
        self.link_triples = link_data

        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.batch_size = batch_size
        # ratio of valid to invalid samples per batch  40
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)

        self.train_indices = np.array(
            list(self.train_triples)).astype(np.int32)
        # These are valid triples, hence all have value 1
        self.train_values = np.array(
            [[1]] * len(self.train_triples)).astype(np.float32)

        self.validation_indices = np.array(
            list(self.validation_triples)).astype(np.int32)
        self.validation_values = np.array(
            [[1]] * len(self.validation_triples)).astype(np.float32)

        # 测试文件
        self.link_indices = np.array(list(self.link_triples)).astype(np.int32)

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)
        self.test_values = np.array(
            [[1]] * len(self.test_triples)).astype(np.float32)

        self.valid_triples_dict = {j: i for i, j in enumerate(
            self.train_triples + self.validation_triples + self.test_triples)}

        print("Total triples count {}, training triples {}, validation_triples {}, test_triples {}".format(
            len(self.valid_triples_dict), len(self.train_indices),
            len(self.validation_indices), len(self.test_indices)))
        # For training purpose
        self.batch_triples = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
        self.batch_labels = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

    def get_iteration_batch(self, iter_num):

        self.tmp_size = self.batch_size
        if (iter_num + 1) * self.batch_size > len(self.train_indices):
            # 这个if专门针对 最后一个批次(不满batch_size)
            last_iter_size = len(self.train_indices) - self.batch_size * iter_num
            self.tmp_size = last_iter_size

        # 每一批次 1个正样本对应 invalid_valid_ratio个负样本
        self.batch_triples = np.empty(
            (self.tmp_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)

        self.batch_labels = np.empty(
            (self.tmp_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

        if (iter_num + 1) * self.batch_size <= len(self.train_indices):
            # 每个批次的 索引
            indices = range(self.batch_size * iter_num,
                            self.batch_size * (iter_num + 1))
        else:
            # 最后一个批次的 索引
            indices = range(self.batch_size * iter_num,
                            len(self.train_indices))

        self.batch_triples[:self.tmp_size, :] = self.train_indices[indices, :]
        self.batch_labels[:self.tmp_size, :] = self.train_values[indices, :]

        last_index = self.tmp_size  # 这个时候batch_triples中已经放好了正样本(行索引tmp_size)

        if self.invalid_valid_ratio > 0:
            # 从0到74085 随机选择若干个索引  作为负样本的实体索引
            random_entities = np.random.randint(
                0, len(self.entity2id), last_index * self.invalid_valid_ratio)

            # np.tile(a,(y,x)) 将a沿着y轴扩大y倍 x轴扩大x倍
            # 将上面的正样本复制了n次填入其中
            self.batch_triples[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_triples[:last_index, :], (self.invalid_valid_ratio, 1))
            self.batch_labels[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_labels[:last_index, :], (self.invalid_valid_ratio, 1))

            # 正式开始造负样本
            for i in range(last_index):
                # 破坏头实体
                for j in range(self.invalid_valid_ratio // 2):
                    current_index = i * (self.invalid_valid_ratio // 2) + j

                    # 破坏头实体，并且判断破坏后的三元组是否是有效三元组【训练集和验证集、测试集】
                    while (random_entities[current_index], self.batch_triples[last_index + current_index, 1],
                           self.batch_triples[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                        random_entities[current_index] = np.random.randint(
                            0, len(self.entity2id))
                    # 对破坏的头实体进行赋值
                    self.batch_triples[last_index + current_index,
                                       0] = random_entities[current_index]
                    self.batch_labels[last_index + current_index, :] = [-1]
                # 破坏尾实体
                for j in range(self.invalid_valid_ratio // 2):
                    current_index = last_index * (self.invalid_valid_ratio // 2) + (
                            i * (self.invalid_valid_ratio // 2) + j)

                    while (self.batch_triples[last_index + current_index, 0],
                           self.batch_triples[last_index + current_index, 1],
                           random_entities[current_index]) in self.valid_triples_dict.keys():
                        random_entities[current_index] = np.random.randint(
                            0, len(self.entity2id))
                    self.batch_triples[last_index + current_index,
                                       2] = random_entities[current_index]
                    self.batch_labels[last_index + current_index, :] = [-1]

            return self.batch_triples, self.batch_labels

        return self.batch_triples, self.batch_labels

    def transe_scoring(self, batch_inputs, entity_embeddings, relation_embeddings):
        source_embeds = entity_embeddings[batch_inputs[:, 0]]
        relation_embeds = relation_embeddings[batch_inputs[:, 1]]
        tail_embeds = entity_embeddings[batch_inputs[:, 2]]
        x = source_embeds - tail_embeds
        x = torch.norm(x, p=1, dim=1)
        return x

    def get_validation_pred(self, args, model):
        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        for iters in range(1):
            start_time = time.time()

            indices = [i for i in range(len(self.test_indices))]
            batch_triples = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))
            entity_list = [j for i, j in self.entity2id.items()]

            ranks_head, ranks_tail = [], []
            reciprocal_ranks_head, reciprocal_ranks_tail = [], []
            hits_at_100_head, hits_at_100_tail = 0, 0
            hits_at_ten_head, hits_at_ten_tail = 0, 0
            hits_at_three_head, hits_at_three_tail = 0, 0
            hits_at_one_head, hits_at_one_tail = 0, 0

            for i in range(batch_triples.shape[0]):
                # print("triple index:", i)
                start_time_it = time.time()
                new_x_batch_head = np.tile(
                    batch_triples[i, :], (len(self.entity2id), 1))
                new_x_batch_tail = np.tile(
                    batch_triples[i, :], (len(self.entity2id), 1))

                new_x_batch_head[:, 0] = entity_list
                new_x_batch_tail[:, 2] = entity_list

                last_index_head = []  # array of already existing triples
                last_index_tail = []
                for tmp_index in range(len(new_x_batch_head)):
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                        new_x_batch_head[tmp_index][2])
                    if temp_triple_head in self.valid_triples_dict.keys():
                        last_index_head.append(tmp_index)

                    temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                        new_x_batch_tail[tmp_index][2])
                    if temp_triple_tail in self.valid_triples_dict.keys():
                        last_index_tail.append(tmp_index)

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them maynot be actually invalid
                new_x_batch_head = np.delete(
                    new_x_batch_head, last_index_head, axis=0)
                new_x_batch_tail = np.delete(
                    new_x_batch_tail, last_index_tail, axis=0)

                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_triples[i], axis=0)
                new_x_batch_tail = np.insert(
                    new_x_batch_tail, 0, batch_triples[i], axis=0)

                import math
                # Have to do this, because it doesn't fit in memory

                if 'WN' in args.dataset:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_head.shape[0] / 4))

                    scores1_head, _ = model(torch.LongTensor(
                        new_x_batch_head[:num_triples_each_shot, :]).cuda())
                    scores2_head, _ = model(torch.LongTensor(
                        new_x_batch_head[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_head, _ = model(torch.LongTensor(
                        new_x_batch_head[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_head, _ = model(torch.LongTensor(
                        new_x_batch_head[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())

                    scores_head = torch.cat(
                        [scores1_head, scores2_head, scores3_head, scores4_head], dim=0)

                else:
                    scores_head, _ = model(new_x_batch_head)

                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)
                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])

                # Tail part here

                if 'WN' in args.dataset:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_tail.shape[0] / 4))

                    scores1_tail, _ = model(torch.LongTensor(
                        new_x_batch_tail[:num_triples_each_shot, :]).cuda())
                    scores2_tail, _ = model(torch.LongTensor(
                        new_x_batch_tail[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_tail, _ = model(torch.LongTensor(
                        new_x_batch_tail[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_tail, _ = model(torch.LongTensor(
                        new_x_batch_tail[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())

                    scores_tail = torch.cat(
                        [scores1_tail, scores2_tail, scores3_tail, scores4_tail], dim=0)


                else:
                    scores_tail, _ = model(new_x_batch_tail)

                sorted_scores_tail, sorted_indices_tail = torch.sort(
                    scores_tail.view(-1), dim=-1, descending=True)

                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_tail.append(
                    np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                # print("ranks_head - ", ranks_head[-1], "    ranks_tail - ", ranks_tail[-1])

            for i in range(len(ranks_head)):
                if ranks_head[i] <= 100:
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:
                    hits_at_one_head = hits_at_one_head + 1

            for i in range(len(ranks_tail)):
                if ranks_tail[i] <= 100:
                    hits_at_100_tail = hits_at_100_tail + 1
                if ranks_tail[i] <= 10:
                    hits_at_ten_tail = hits_at_ten_tail + 1
                if ranks_tail[i] <= 3:
                    hits_at_three_tail = hits_at_three_tail + 1
                if ranks_tail[i] == 1:
                    hits_at_one_tail = hits_at_one_tail + 1

            assert len(ranks_head) == len(reciprocal_ranks_head)
            assert len(ranks_tail) == len(reciprocal_ranks_tail)
            print("here {}".format(len(ranks_head)))
            print("\nCurrent iteration time {}".format(time.time() - start_time))

            average_hits_at_100_head.append(
                hits_at_100_head / len(ranks_head))
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

            average_hits_at_100_tail.append(
                hits_at_100_tail / len(ranks_head))
            average_hits_at_ten_tail.append(
                hits_at_ten_tail / len(ranks_head))
            average_hits_at_three_tail.append(
                hits_at_three_tail / len(ranks_head))
            average_hits_at_one_tail.append(
                hits_at_one_tail / len(ranks_head))
            average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
            average_mean_recip_rank_tail.append(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

        print("\nAveraged stats for replacing head are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
        print("Mean rank {}".format(
            sum(average_mean_rank_head) / len(average_mean_rank_head)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))

        print("\nAveraged stats for replacing tail are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)))
        print("Mean rank {}".format(
            sum(average_mean_rank_tail) / len(average_mean_rank_tail)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))

        cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                               + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
        cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                               + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
        cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                                 + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
        cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                               + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
        cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                                + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
        cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
            average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

        print("\nCumulative stats are -> ")
        print("Hits@100 are {}".format(cumulative_hits_100))
        print("Hits@10 are {}".format(cumulative_hits_ten))
        print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))

        return cumulative_mean_recip_rank, cumulative_mean_rank, cumulative_hits_one, cumulative_hits_three, cumulative_hits_ten

    def get_validation_pred2(self, args, model):

        start_time = time.time()

        indices = [i for i in range(len(self.link_indices))]

        batch_triples = self.link_indices[indices, :]  # (64705, 3)

        print("link set length : ", len(self.link_indices))

        entity_list = [j for i, j in self.entity2id.items()]

        sort_list = []

        for i in range(batch_triples.shape[0]):
            start_time_it = time.time()
            # batch_triples [[38814     6    -1],..]
            # 扩展当前要预测的三元组(74085, 3)
            new_x_batch_head = np.tile(
                batch_triples[i, :], (len(self.entity2id), 1))

            if new_x_batch_head[0][0] == -1:
                new_x_batch_head[:, 0] = entity_list
            elif new_x_batch_head[0][2] == -1:
                new_x_batch_head[:, 2] = entity_list

            # 删除训练集中已有的三元组
            # last_index_head = []  # array of already existing triples
            #
            # for tmp_index in range(len(new_x_batch_head)):
            #     temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
            #                         new_x_batch_head[tmp_index][2])
            #     if temp_triple_head in self.valid_triples_dict.keys():
            #         last_index_head.append(tmp_index)
            #
            # new_x_batch_head = np.delete(
            #     new_x_batch_head, last_index_head, axis=0)

            # head part here
            scores_head, _ = model.test(new_x_batch_head)

            sorted_scores_head, sorted_indices_head = torch.sort(
                scores_head.view(-1), dim=-1, descending=True)

            sorted_indices = sorted_indices_head[:10].cpu().numpy()

            sort_list.append(sorted_indices)
            print("it:{},time:{}".format(i, time.time() - start_time_it))

        print("链接预测总共用的时间:{}".format(time.time() - start_time))

        return sort_list
