import torch
import dataloader
import utils
import torch.nn as nn
import yaml
import os
import pathlib
import plotting
from datetime import datetime

main_path = pathlib.Path(__file__
                         )
with open(os.path.join(main_path.parents[0], 'config.yaml')) as file:
    config = yaml.safe_load(file)
data_config = config['Dataloader']
plot_config = config['Plotting']


dataset_num = data_config['dataset_num']


if dataset_num == 'FD001':
            window_length = 30
            in_feat_multiplier = 18
            eng_list = plot_config['First']
elif dataset_num == 'FD002':
            window_length = 20
            in_feat_multiplier = 8
            eng_list = plot_config['Second']
elif dataset_num == 'FD003':
            window_length = 38
            in_feat_multiplier = 26
            eng_list = plot_config['Third']
elif dataset_num == 'FD004':
            window_length = 19
            in_feat_multiplier = 6  
            eng_list = plot_config['Fourth']
            
subset = 'FD003'
timestamp = '2023-01-29_20-48-03'
net = 'LSTM'
trial_num = 'Num_Trial 123'
batch_size = 832
            
train_data , test_data , train_lengths , test_lengths, total_iter = dataloader.train_test_loader(
                                                                                                batch_size = batch_size, window_length=19)
c_time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


result_path = os.path.join(main_path.parents[0],'Transfer Learning','Results',net + 'Fixed' ,subset + '-' + dataset_num,c_time_stamp,trial_num)
if not os.path.exists(result_path):
    os.makedirs(result_path)
    
models = torch.load(os.path.join(main_path.parents[1],'Best study and trials',net,subset,timestamp,trial_num,'Model.pt'))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 1
criterion = nn.MSELoss()
models.eval()
with torch.no_grad():
    test_loss = 0
    total_iterations = 0
    for i, (data,label) in enumerate(test_data):
        data,label = data.to(device), label.to(device)
        pred = models(data)
        loss = torch.sqrt(criterion(pred,label))
        pred = pred.detach().cpu()
        label = label.detach().cpu()
        test_loss+=loss.detach().item()
        if i == 0:    
            predictions = pred
            targets = label
        else:
            predictions = torch.cat((predictions,pred),dim = 0)
            targets = torch.cat((targets,label),dim = 0)
        total_iterations += 1
    total_test_loss = test_loss/total_iterations

test_score = utils.score_cal(test_preds= predictions,
                             test_labels = targets,
                             test_lengths = test_lengths,
                             test_loss = total_test_loss,
                             epoch = epoch,
                             result_path = result_path,
                             testing = False)

plot = plotting.train_test_plots(dataset_num = dataset_num, 
                                 result_path= result_path,
                                 train_lengths= 2,
                                 test_lengths= test_lengths,
                                 eng_list = eng_list
                                         )
plot.testing_plots(preds = predictions, 
                   labels = targets, 
                   n_epoch=epoch, 
                   )
print(test_score)

