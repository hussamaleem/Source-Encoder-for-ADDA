import optuna
import torch
import model
import dataloader
import train_test
import torch.nn as nn
import utils
from optuna.trial import TrialState
import os
import plotting
import torch.optim as optim
import pathlib
import yaml
from datetime import datetime
from tqdm import trange




main_path = pathlib.Path(__file__
                         )
with open(os.path.join(main_path.parents[0], 'config.yaml')) as file:
    config = yaml.safe_load(file)
    optuna_config = config['Optuna']
    training_config = config['Training']
    data_config = config['Dataloader']
    plot_config = config['Plotting']

class OptunaOptim():
    
    def __init__(self,
                 n_epochs = training_config['epochs'],
                 n_trials = optuna_config['n_trials'],
                 start_test_epoch = training_config['waiting_epochs'],
                 interval = training_config['interval'],
                 dataset_num = data_config['dataset_num'] ,
                 ):

        self.n_epochs = n_epochs
        self.n_trials = n_trials
        self.start_test_epoch = start_test_epoch
        self.interval = interval
        self.dataset_num = dataset_num
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.best_loss_list = []
        self.best_epoch_list = []
        
        if self.dataset_num == 'FD001':
            self.eng_list = plot_config['First']
            self.window_length = 30
            self.in_feat_multiplier = 18
        elif self.dataset_num == 'FD002':
            self.window_length = 20
            self.in_feat_multiplier = 8
            self.eng_list = plot_config['Second']
        elif self.dataset_num == 'FD003':
            self.window_length = 38
            self.in_feat_multiplier = 26
            self.eng_list = plot_config['Third']
        elif self.dataset_num == 'FD004':
            self.window_length = 19
            self.in_feat_multiplier = 6
            self.eng_list = plot_config['Fourth']
        

       
    def create_parameters(self,trial,lstm):
        
        self.batch_size = trial.suggest_int('BatchSize', optuna_config['batch_size_low'],
                                                         optuna_config['batch_size_high'], 
                                                         step = optuna_config['batch_size_step'])
        
        self.lr = trial.suggest_float('lr', optuna_config['learning_rate_low'],
                                            optuna_config['learning_rate_high'], log=True)
        
        self.initializer = trial.suggest_categorical('initializer',[optuna_config['initializer_1'],
                                                                    optuna_config['initializer_2'],
                                                                    optuna_config['initializer_3'],
                                                                    optuna_config['initializer_4']])
        
        self.out_features = trial.suggest_int('Out_Features', optuna_config['out_feature_low'],
                                                              optuna_config['out_feature_high'], 
                                                              step = optuna_config['out_feature_step'])
        

        if lstm:
            
            self.hidden_size = trial.suggest_int('Hidden_Size', optuna_config['hidden_size_low'],
                                                             optuna_config['hidden_size_high'], 
                                                             step = optuna_config['hidden_size_step'])
            
        else:
            
            self.out_channel = trial.suggest_int('Out_Channel', optuna_config['out_channel_low'],
                                                          optuna_config['out_channel_high'], 
                                                          step = optuna_config['out_channel_step'])
            

        
    def objective(self,trial):
        
        num = trial.number
        test_score_list = []
        total_training_loss = []
        lr_list = []
        training_epoch_list = []
        self.test_loss_list = []

        self.create_parameters(trial, lstm= True)
        
        self.train_data , self.test_data , self.train_lengths , self.test_lengths, self.total_iter = dataloader.train_test_loader(
                                                                                    batch_size = self.batch_size,
                                                                                    window_length = self.window_length)
        
        print(f'Batch_size{self.batch_size}')
        print(f'hidden_size{self.hidden_size}')
        models = model.LSTM_Model( self.hidden_size, out_feature = self.out_features, in_channel = 14).to(self.device)
        print(f'Model: {models.__class__.__name__}')
        optimizer = optim.Adam(models.parameters(), lr = self.lr, weight_decay=1e-3)

        criterion = nn.MSELoss()
        lr_sched = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
        models.weight_initialization(self.initializer)
        
        self.cd = pathlib.Path().cwd()
        self.result_path = os.path.join(self.cd,'00_Results', 
                                        models.__class__.__name__, 
                                        self.dataset_num , 
                                        self.time_stamp,
                                        'Num_Trial ' + str(num))
        
        self.result_excel = pathlib.Path(self.result_path).parent
        self.result_excel = os.path.join(self.result_excel, ('Detail Result Files'))
        if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)
        if not os.path.exists(self.result_excel):
            os.mkdir(self.result_excel)
            
        train_test_class = train_test.train_test_loops(models = models, 
                                                       device = self.device, 
                                                       optimizer = optimizer, 
                                                       criterion = criterion,
                                                       )

        plot = plotting.train_test_plots(dataset_num = self.dataset_num, 
                                         result_path=self.result_path,
                                         train_lengths=self.train_lengths,
                                         test_lengths=self.test_lengths,
                                         eng_list = self.eng_list
                                         )

        print('..........Initiating Training Loop.......') 
        pbar = trange(self.n_epochs)              
        
        for epoch in pbar:

            pbar.set_description(f'Epoch {epoch}')
            training_epoch_list.append(epoch)
            training_preds , training_targs, training_loss = train_test_class.training_loop(train_data = self.train_data)
            
            total_training_loss.append(training_loss)
            pbar.set_postfix(Training_Loss = training_loss)
            lr_sched.step()
            current_lr = (lr_sched.get_last_lr())
            lr_list.append(float(current_lr[0]))
            
            if ((epoch+1) > self.start_test_epoch) and ((epoch+1)%self.interval == 0):
                
                print('\n.........Initializing Testing Loop........')
                
                test_pred , test_targ, test_loss = train_test_class.testing_loop(test_data = self.test_data)

                plot.testing_plots(preds = test_pred, 
                                   labels = test_targ, 
                                   n_epoch=epoch, 
                                   )
                
                plot.training_rul_plots(preds = training_preds, 
                                        labels=training_targs, 
                                        n_epoch=epoch, 
                                        )

    
                test_score = utils.score_cal(test_preds=test_pred,
                                             test_labels = test_targ,
                                             test_lengths = self.test_lengths,
                                             test_loss = test_loss,
                                             epoch = epoch,
                                             result_path = self.result_path,
                                             testing = False)
                

                pbar.set_postfix(Training_loss = training_loss, 
                                 Test_loss = test_loss,
                                 Test_score = test_score,
                                 )
                
                self.test_loss_list.append(test_loss)
                test_score_list.append(test_score)
                
                utils.model_saver(n_epoch = epoch, 
                                  cd = self.cd,  
                                  n_trial = num, 
                                  time_stamp = self.time_stamp,
                                  start_epoch = self.start_test_epoch, 
                                  model = models,  
                                  score_val = test_score
                                  )
                
                trial.report(test_score, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        
        self.best_loss_list.append(utils.best_loss)
        self.best_epoch_list.append(utils.best_epoch) 
           
        utils.learning_rate_excel(lr_list = lr_list, 
                                  epoch_list = training_epoch_list , 
                                  loss_list = total_training_loss,
                                  result_path = self.result_path
                                  )
        
        plot.loss_plots(loss = total_training_loss, train = True)
        plot.loss_plots(loss = self.test_loss_list, train = False)
        
        utils.test_score_list.clear()
        utils.saver_list.clear()
        
        return min(test_score_list)
    
    def run_objective(self):
        
        sampler = optuna.samplers.TPESampler(n_startup_trials= 20, constant_liar= True)
        self.study = optuna.create_study(direction = 'minimize', sampler = sampler)
        self.study.optimize(self.objective, n_trials=self.n_trials, gc_after_trial=True)
    
    
    def create_summary(self):
        
        fig = optuna.visualization.plot_param_importances(self.study)
        fig.show(renderer = 'browser')
        fig.write_image(os.path.join(self.result_excel,'Hyperparameter Importance.jpeg'))
        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])   
        print('\nStudy Summary')
        print(f'Number of finished trials: {len(self.study.trials):2}')
        print(f'Number of pruned trials: {len(pruned_trials):2}')
        print(f'Number of completed trials: {len(complete_trials):2}')
        best_trial = self.study.best_trial
        print("Best trial:")
        print("  Value: ", best_trial.value)
        print("  Params: ")
        results_df = self.study.trials_dataframe()
        utils.param_results(best_trial = best_trial, result_excel = self.result_excel)
        utils.results_dataframe(results_df = results_df, 
                                result_excel = self.result_excel, 
                                best_epoch = self.best_epoch_list,
                                best_loss = self.best_loss_list)

        
        
        
        
