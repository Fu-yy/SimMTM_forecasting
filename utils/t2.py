def forward(self, similarity_matrix, batch_emb_om):
    """
    similarity_matrix: 样本之间的相似性矩阵
    batch_emb_om: 当前批次样本的表示向量
    """
    cur_batch_shape = batch_emb_om.shape

    # 计算加权相似性矩阵
    similarity_matrix /= self.temperature
    similarity_matrix = similarity_matrix - torch.eye(cur_batch_shape[0]).to(similarity_matrix.device) * 1e12

    # 生成权重矩阵
    rebuild_weight_matrix = self.softmax(similarity_matrix)

    # 按相似性权重加权聚合
    rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om.reshape(cur_batch_shape[0], -1))

    # 分离重建的原始序列
    rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1], -1)

    return rebuild_weight_matrix, rebuild_oral_batch_emb
