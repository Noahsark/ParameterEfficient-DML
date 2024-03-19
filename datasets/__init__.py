import datasets.cars196
import datasets.cub200
import datasets.stanford_online_products
import datasets.inshop
import datasets.inaturalist


def select(opt, **kwargs):
    dataset = opt.dataset
    data_path = opt.source_path
    
    if 'cub200' in dataset:
        return datasets.cub200.give_dataloaders(opt, data_path)
    elif 'cars196' in dataset:
        return datasets.cars196.give_dataloaders(opt, data_path)
    elif ('online_products' in dataset) or ('sop' in dataset):
        return datasets.stanford_online_products.give_dataloaders(opt, data_path)
    elif 'inshop' in dataset:
        return datasets.inshop.give_dataloaders(opt, data_path)
    elif 'inaturalist' in dataset:
        return datasets.inaturalist.give_dataloaders(opt, data_path)
    else:
        raise NotImplementedError(
            'A dataset for {} is currently not implemented.'
            .format(dataset))
