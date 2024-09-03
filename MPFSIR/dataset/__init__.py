def create_datasets(config, **args):

    if config.dataset == "3dpw":
        from dataset.dataset_3dpw import create_datasets as create_3dpw_datasets
        return create_3dpw_datasets(config, **args)
    
    elif config.dataset == "handball_shot":
        from dataset.dataset_handball_shot import create_datasets as create_handball_shot_datasets
        return create_handball_shot_datasets(config, **args)