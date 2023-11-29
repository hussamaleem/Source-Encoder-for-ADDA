import os
import pandas as pd
import torch
import numpy as np
import sklearn.preprocessing as preprocessor
import pathlib
import xlsxwriter
import yaml

main_path = pathlib.Path(__file__
                         )
with open(os.path.join(main_path.parents[0], 'config.yaml')) as file:
    config = yaml.safe_load(file)
    data_config = config['Dataloader']

dataset_num = data_config['dataset_num']
saver_list = []
test_score_list = []
best_epoch = 0
best_loss = 1

def load_data(
              dataset_path: str,
              task: str,
              dataset_num: str
              ):
        
        
        drop_columns = ['s1','s5','s6','s10','s16','s18','s19']
        file = pd.read_csv(os.path.join(dataset_path,
                                        f'{task}_{dataset_num}.csv')).drop(['setting1',
                                                                             'setting2',
                                                                             'setting3'],axis=1)
        
        file_list = []
        cycle_list = []
        grouped_file = file.groupby(file.engine_id)
        total_eng = file['engine_id'].iloc[-1]
        file = file.drop(drop_columns,axis=1, inplace=True)
        for i in range(total_eng):
            i = i+1
            file_grouped = grouped_file.get_group(i)
            cycle_list.append(file_grouped['cycle'].iloc[-1])
            file_list.append(file_grouped.drop(['engine_id','cycle'],axis=1))
    
        return file_list , cycle_list, total_eng
    
def labels_generator(
                     cycle_list: list,
                     window_length: int,
                     train: bool,
                     data_length,
                     scaling_method: str,
                     dataset_path,
                     total_eng
                     ):

        labels = []
        ext_rul = pd.read_csv(os.path.join(dataset_path, f'RUL_{dataset_num}.csv')).values
        start_rul = 120
        for i in range(total_eng):
            
            
            if train:
                total_rul = cycle_list[i]
            else:
                total_rul = cycle_list[i] + ext_rul[i].item()
                
                
            healthy_cyc = total_rul - start_rul
            
            if not train:
                print(f'Total_RUL:{total_rul} && Start_Rul:{start_rul}')
                print(healthy_cyc)
            degrading_cyc = total_rul - healthy_cyc
            
            healthy_rul = start_rul * np.ones((healthy_cyc,1))
            
            degrading_rul = np.linspace(start = start_rul -1 ,
                                        stop = 0,
                                        num = degrading_cyc)
            
            degrading_rul = np.expand_dims(degrading_rul, axis = 1)
            
            final_rul = np.concatenate((healthy_rul,degrading_rul), axis = 0)
            
            if train:
                final_rul = final_rul
            else:
                final_rul = final_rul[:cycle_list[i],:]
                
            if data_length[i]<final_rul.shape[0]:
                if (train == False and ext_rul[i].item()>=start_rul) : 
                    
                    final_rul = final_rul[:data_length[i]] 
                else:
                    
                    final_rul = final_rul[-data_length[i]:]
                
            labels.append(torch.Tensor(final_rul))
            
        return torch.cat(labels,dim=0)
    
def scale_data(
               data,
               data_scale_params,
               scaling_method
               ):
        
        if scaling_method == '-11':
            scaler = preprocessor.MinMaxScaler(feature_range = (-1,1))
        elif scaling_method == '01':
            scaler = preprocessor.MinMaxScaler(feature_range = (0,1))
        else:
            scaler = preprocessor.StandardScaler()
        
        if data_scale_params is None:
            data_scale_params = scaler.fit(pd.concat(data , axis = 0))

        scaled_data = [torch.FloatTensor(data_scale_params.transform(x)) for x in data]
        
        return scaled_data, data_scale_params

def window_data(
                dataset,
                window_length,
                window_step
                ):
        
        windowed_data_list = []
        
        for data in dataset:
            
            windowed_data_list.append(data.unfold(0,window_length,window_step))
            
        eng_id_lengths = [i.shape[0] for i in windowed_data_list]
        
        return torch.cat(windowed_data_list, dim = 0), eng_id_lengths
    
def score_cal(
              test_preds,
              test_labels,
              test_lengths,
              epoch,
              test_loss,
              result_path,
              testing
              ):
    
        test_preds = torch.split(test_preds,test_lengths)
        test_labels = torch.split(test_labels,test_lengths) 
        score_list = []
        pred_list = []
        label_list = []
        a1 = 13
        a2 = 10
        
        for i in range(len(test_preds)):
            
            pred_RUL = test_preds[i][-1] 
            actual_RUL = test_labels[i][-1]
            pred_list.append(pred_RUL)
            label_list.append(actual_RUL)
            d = pred_RUL - actual_RUL
            
            if d < 0:
                
                score_list.append((torch.exp(-(d/a1)) - 1).item())
                
            else:
                
                score_list.append((torch.exp((d/a2)) - 1).item())
                
        test_score_list.append(sum(score_list))
        
        if not testing:
            if sum(score_list) <= min(test_score_list):
                accuracy_excel(test_score_list = score_list, 
                               epoch = epoch, 
                               test_loss = test_loss, 
                               prediction_list = pred_list, 
                               target_list = label_list, 
                               sum_score = sum(score_list), 
                               result_path = result_path)
            
                global best_epoch 
                global best_loss
            
                best_epoch = epoch
                best_loss = test_loss

        return sum(score_list)
        
def param_results(best_trial,result_excel):
        
        best_para = xlsxwriter.Workbook(os.path.join(result_excel, f'Best Parameters (Trial num {best_trial.number}).xlsx'))
        worksheet = best_para.add_worksheet()
        bold = best_para.add_format({'bold': True})
        worksheet.write('A1', 'Hyperparameters',bold)
        worksheet.write('B1', 'Value',bold)
        col_num = 1
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))
            worksheet.write(col_num,0, key)
            worksheet.write(col_num,1, value)
            col_num = col_num+1
        best_para.close()
        print('Best Paramters file exported')
        
def results_dataframe(results_df, result_excel, best_epoch, best_loss):
        
        converter = pd.ExcelWriter(os.path.join(result_excel, 'Result Dataframe.xlsx'))
        converter_2 = pd.ExcelWriter(os.path.join(result_excel, 'Result Dataframe Sorted.xlsx'))
        results_df.to_excel(converter)
        converter.save()
        results_df = results_df.drop(results_df[results_df.state == 'PRUNED'].index)
        results_df['Test Loss'] = best_loss
        results_df['Epoch'] = best_epoch
        results_df = results_df.drop(['datetime_start','datetime_complete', 'duration'], axis=1)
        results_df = results_df.sort_values(by = 'value', ascending=True)
        results_df.to_excel(converter_2)
        converter_2.save()
        print('Results Dataframe exported')

def model_saver(n_epoch,cd ,n_trial ,time_stamp , start_epoch, model, score_val):
        
        sav_path = os.path.join(cd,'00_Checkpoint', dataset_num, time_stamp , 'Num_trial ' + str(n_trial))
        if not os.path.exists(sav_path):
                    os.makedirs(sav_path)
        
        
        if n_epoch == (start_epoch+9):
            
            torch.save(model.state_dict(), sav_path +  '/Model Parameters')
            torch.save(model, sav_path + '/Model.pt')
            saver_list.append(score_val)
            
        elif (n_epoch > (start_epoch+9)) and (score_val < min(saver_list)):
            
            torch.save(model.state_dict(),sav_path + '/Model Parameters')
            torch.save(model, sav_path + '/Model.pt')
            saver_list.append(score_val)
            
def learning_rate_excel(lr_list,epoch_list,loss_list, result_path):
        
        row_num = 1
        lr_workbook = xlsxwriter.Workbook(os.path.join(result_path, 'Learning Rates.xlsx'))
        worksheet = lr_workbook.add_worksheet()
        bold = lr_workbook.add_format({'bold': True})
        worksheet.write('A1', 'Epochs',bold)
        worksheet.write('B1', 'Learning Rate',bold)
        worksheet.write('C1', 'Training Loss',bold)
        for value in range(len(epoch_list)):
            worksheet.write(row_num,0, epoch_list[value])
            worksheet.write(row_num,1, lr_list[value])
            worksheet.write(row_num,2, loss_list[value])
            row_num = row_num+1
        lr_workbook.close()
        
def accuracy_excel(test_score_list,
                   epoch, 
                   test_loss, 
                   prediction_list, 
                   target_list, 
                   sum_score,
                   result_path):
            
            accuracy_workbook = xlsxwriter.Workbook(os.path.join(result_path, 
                                                                 'Accuracy Scores.xlsx'))
            worksheet = accuracy_workbook.add_worksheet()
            formating = accuracy_workbook.add_format({'bold': True,
                                                   'align': 'center',
                                                   'border': 2})
            worksheet.write('A1', 'Epochs',formating)
            worksheet.write('B1', 'Test Loss', formating)
            worksheet.write('C1', 'Sum Score' , formating)
            worksheet.write('D1', 'Engine Id' , formating)
            worksheet.write('E1', 'Prediction' , formating)
            worksheet.write('F1', 'Label' , formating)
            worksheet.write('G1', 'Engine Scores' , formating)
            worksheet.write('A2', epoch)
            worksheet.write('B2', test_loss)
            worksheet.write('C2', sum_score)
            for i in range(len(prediction_list)):
                worksheet.write(i+1,3,i+1)
                worksheet.write(i+1,4,prediction_list[i])
                worksheet.write(i+1,5,target_list[i])
                worksheet.write(i+1,6,test_score_list[i])
            accuracy_workbook.close()
        

