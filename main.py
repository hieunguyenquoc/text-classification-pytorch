import torch
from src.preprocess import Preprocessing
from src.model import TextClassification
from src.parser_data import parameter_parser
# import numpy as np
from fastapi import FastAPI
import uvicorn

class Inference:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda:0"
            print("Run on GPU")
        else:
            self.device = "cpu"
            print("Run on CPU")
        
        args = parameter_parser()
        self.model = TextClassification(args)
        self.model.load_state_dict(torch.load("model/model_RNN_Glove.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = Preprocessing(args)
        self.preprocess.load_data()
        self.preprocess.tokenization()

    def predict(self,sentence):
        # prediction = []
        sentence_to_pred = self.preprocess.sequence_to_token(sentence)
        sentence_tensor = torch.from_numpy(sentence_to_pred)
        sentence_pred = sentence_tensor.type(torch.LongTensor)
        sentence_pred = sentence_pred.to(self.device)
        y_pred = self.model(sentence_pred)
        final_pred = y_pred.cpu().detach().numpy()
        return final_pred

app = FastAPI()

result = Inference()
@app.post("/text_classification")
def text_classification(sentence:str):
   
    result_numpy = result.predict([sentence])
    if result_numpy < 0.5:
        return "positive"
    else:
        return "negative"

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)


        

    