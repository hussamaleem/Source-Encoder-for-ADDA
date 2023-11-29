import torch


class train_test_loops():
    
    def __init__(self,
                 models,
                 device,
                 optimizer,
                 criterion,
                 ):
        self.models = models
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        
        '''
        Class for implenting training and testing loops
        
        Args:
            
        self.models : Instance of the LSTM Network
        self.criterion : Loss function (MSE)
        self.device : device for compuatation (CPU or GPU) 
        self.sclaer : scaler fucntion used to fit the labels, will be used to inverse
                        transform predictions and labels
        Returns:

        Training Loop;
        
        Preictions: Concatenated training predictions per epoch
        Targets : Concatenated training labels per epoch               
        Total_loss : Average training loss of iterations per epoch
        
        Testing Loop:
            
        Preictions: Concatenated testing predictions per epoch
        Targets : Concatenated testing labels per epoch               
        Total_test_loss : Average test loss of iterations per epoch
        
        
        '''
        
    def training_loop(self,train_data):
        train_loss = 0
        total_iterations = 0
        self.models.train()

        for i,(data,label) in enumerate(train_data):
            self.optimizer.zero_grad()
            data,label = data.to(self.device), label.to(self.device)
            pred= self.models(data)
            loss = torch.sqrt(self.criterion(pred,label))
            loss.backward()
            self.optimizer.step()
            pred = pred.detach().cpu()
            label = label.detach().cpu()
            train_loss += loss.detach().item()
            if i == 0:    
                predictions = pred
                targets = label
            else:
                predictions = torch.cat((predictions,pred),dim = 0)
                targets = torch.cat((targets,label),dim = 0)

            total_iterations += 1
        total_loss = train_loss/total_iterations
    
        return predictions , targets , total_loss

    def testing_loop(self,test_data):
        
        self.models.eval()
        with torch.no_grad():
            test_loss = 0
            total_iterations = 0
            for i, (data,label) in enumerate(test_data):
                data,label = data.to(self.device), label.to(self.device)
                pred = self.models(data)
                loss = torch.sqrt(self.criterion(pred,label))
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
            return predictions, targets, total_test_loss

