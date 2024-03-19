
import copy
import os
from datasets.basic_dataset import BaseDataset


def give_dataloaders(opt, datapath):
    
    image_sourcepath = datapath
    train_image_dict, val_image_dict  = {},{}
    with open(os.path.join(image_sourcepath,'Inaturalist_train_set1.txt')) as f:
        FileLines = f.readlines()
        FileLines = [x.strip() for x in FileLines]
        for entry in FileLines:
            info = entry.split('/')
            if '/'.join([info[-3],info[-2]]) not in train_image_dict:
                train_image_dict['/'.join([info[-3],info[-2]])] = []
            train_image_dict['/'.join([info[-3],info[-2]])].append(os.path.join(image_sourcepath,entry))
            
    with open(os.path.join(image_sourcepath,'Inaturalist_test_set1.txt')) as f:
        FileLines = f.readlines()
        FileLines = [x.strip() for x in FileLines]
        for entry in FileLines:
            info = entry.split('/')
            if '/'.join([info[-3],info[-2]]) not in val_image_dict:
                val_image_dict['/'.join([info[-3],info[-2]])] = []
            val_image_dict['/'.join([info[-3],info[-2]])].append(os.path.join(image_sourcepath,entry))
    new_train_dict = {}
    class_ind_ind = 0
    for cate in train_image_dict:
        new_train_dict[class_ind_ind] = train_image_dict[cate]
        class_ind_ind += 1
    train_image_dict = new_train_dict
    
    new_val_dict = {}
    class_ind_ind = 0
    for cate in val_image_dict:
        new_val_dict[class_ind_ind] = val_image_dict[cate]
        class_ind_ind += 1
    val_image_dict = new_val_dict
    
    train_dataset       = BaseDataset(train_image_dict, opt)
    val_dataset         = BaseDataset(val_image_dict,   opt, is_validation=True)
    eval_dataset        = BaseDataset(train_image_dict, opt, is_validation=True)
    
    print(
        '\nDataset Setup: \n#Classes: Train ({0}) | Val ({1}) | Test ({2})\n'.
        format(len(train_image_dict),
               len(val_image_dict) if val_image_dict is not None else 'X',
               len(val_image_dict)))
    
    return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}