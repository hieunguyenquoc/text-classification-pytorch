import torch
from preprocess import Preprocessing
from model import TextClassification
from parser_data import parameter_parser
import numpy as np
from fastapi import FastAPI
import uvicorn

class Inference:
    def __init__(self):
        args = parameter_parser()
        self.model = TextClassification(args)
        self.model = torch.load("model/model_RNN_Glove.ckpt", map_location=torch.device('cpu'))
        self.model.eval()
        self.preprocess = Preprocessing(args)
        self.preprocess.load_data()
        self.preprocess.tokenization()

    def predict(self,sentence):
        # prediction = []
        sentence_to_pred = self.preprocess.sequence_to_token(sentence)
        sentence_tensor = torch.from_numpy(sentence_to_pred)
        sentence_pred = sentence_tensor.type(torch.LongTensor)
        y_pred = self.model(sentence_pred)
        final_pred = y_pred.detach().numpy()
        return final_pred

app = FastAPI()

@app.post("/text_classification")
def text_classification(sentence:str):
    result = Inference()
    result_numpy = result.predict([sentence])
    if result_numpy < 0.5:
        return "positive"
    else:
        return "negative"

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)


        

    