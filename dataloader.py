import dataset
import os
import pathlib
import yaml

from torch.utils.data import DataLoader

main_path = pathlib.Path(__file__
                         )
with open(os.path.join(main_path.parents[0], 'config.yaml')) as file:
    config = yaml.safe_load(file)
    data_config = config['Dataloader']

def train_test_loader(batch_size, window_length):
    
    train_dataset  = dataset.CmapssDataset(
                                root = main_path.parents[1], 
                                train = True,
                                dataset_num = data_config['dataset_num'],
                                window_step = data_config['window_step'],
                                data_scale_params = None,
                                scaling_method = data_config['scaling_method'],
                                bsc = data_config['bsc'],
                                window_length = window_length
                                        )
    
    test_dataset = dataset.CmapssDataset(
                                root = main_path.parents[1], 
                                train = False,
                                dataset_num = data_config['dataset_num'],
                                window_step = data_config['window_step'],
                                data_scale_params = train_dataset.data_scale_params,
                                scaling_method = data_config['scaling_method'],
                                bsc = data_config['bsc'],
                                window_length = window_length
                                        )
    
    train_loader = DataLoader(train_dataset,batch_size = batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

    train_lengths = train_dataset.eng_id_lengths
    test_lengths = test_dataset.eng_id_lengths
    total_iter = len(train_loader)

    return train_loader,test_loader,train_lengths,test_lengths,total_iter