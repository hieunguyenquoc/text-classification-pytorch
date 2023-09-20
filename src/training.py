import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from preprocess import Preprocessing
from model import TextClassification
import torch.nn.functional as F
from parser_data import parameter_parser
import numpy as np

SEED = 2019
torch.manual_seed(SEED)

class DataMapper(Dataset): #DataSet provied an way to iterate by batch(handle data by batch)
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
#Excuate training phase
class Exucuate:
    def __init__(self,args):
        self.__init_data__(args)
        if torch.cuda.is_available():
            self.device = "cuda:0"
            print("Run on GPU")
        else:
            self.device = "cpu"
            print("Run on CPU")
        

        self.args = args
        self.batch_size = args.batch_size

        self.model = TextClassification(args)
        self.model.to(self.device)
    #define preprocess data
    def __init_data__(self,args):

        self.preprocess = Preprocessing(args)
        self.preprocess.load_data()
        self.preprocess.tokenization()

        #load data from csv
        raw_x_train = self.preprocess.X_train
        raw_x_test = self.preprocess.X_test

        #preprocess the data
        self.x_train = self.preprocess.sequence_to_token(raw_x_train)
        self.x_test = self.preprocess.sequence_to_token(raw_x_test)

        self.y_train = self.preprocess.Y_train
        self.y_test = self.preprocess.Y_test

    def train(self):
        
        #load data to DataMapper
        training_set = DataMapper(self.x_train, self.y_train)
        #print(type(training_set))
        test_set = DataMapper(self.x_test, self.y_test)
        
        
        #load data to Dat
        self.load_train = DataLoader(training_set, batch_size=self.batch_size)
        self.load_test = DataLoader(test_set)

        #define optimize 
        optimizer = optim.RMSprop(self.model.parameters(), lr=args.learning_rate)
        print(self.model)

        #set model to cpu or gpu
        
        #train procedure
        for epoch in range(args.epochs):
            
            #define prediction
            prediction = []

            #model at train mode
            self.model.train()

            #train per batch
            for x_batch, y_batch in self.load_train:
                #get data type to tensor
                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.FloatTensor).unsqueeze(1)
                x = x.to(self.device)
                y = y.to(self.device)
                #train
                y_pred = self.model(x)
                
                #loss function
                loss = F.binary_cross_entropy(y_pred, y)
                
                #optimizer
                optimizer.zero_grad()

                #calculatte backpropagation
                loss.backward()

                optimizer.step()

                prediction += list(y_pred.cpu().squeeze().detach().numpy())
        #test prediction result
        torch.save(self.model.state_dict(),"model/model_RNN_Glove.pt")

        test_prediction = self.evaluation()

        train_accuaracy = self.calculate_accuracy(self.y_train, prediction)
        test_accuaracy = self.calculate_accuracy(self.y_test, test_prediction)

        #save model to cpkt file
       

        print("Epoch : %d, loss : %.5f, Train accuaracy :%.5f, Test accuracy: %.5f" % (epoch + 1, loss.item(), train_accuaracy, test_accuaracy))

    def evaluation(self):

        prediction = []
        self.model.eval()

        with torch.no_grad():
            for x_batch, y_batch in self.load_test:
                
                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.FloatTensor)
                x = x.to(self.device)
                y = x.to(self.device)
                print(np.shape(x))
                y_pred = self.model(x)
                prediction += list(y_pred.cpu().detach().numpy())
        #print(prediction)
        return prediction
    
    @staticmethod
    def calculate_accuracy(ground_true, predictions):

        true_positive = 0
        true_negative = 0

        for true, pred in zip(ground_true, predictions):
            if pred >= 0.5 and (true == 1):
                true_positive += 1
            elif pred < 0.5 and true == 0:
                true_negative += 1
        return (true_positive + true_negative) / len(ground_true)

if __name__ == "__main__":
    args = parameter_parser()
    excute = Exucuate(args)
    excute.train()





        
