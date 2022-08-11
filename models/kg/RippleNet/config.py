class Config:
    dataset = 'movie'
    embed_dim = 16
    n_hop = 2
    kge_weight = 0.01
    l2_weight = 1e-7
    lr = 0.001
    batch_size = 1024
    epochs = 10
    n_memory = 32
    item_update_mode = 'plus_transform'  # 不同mode对训练影响很大
    using_all_hops = True
