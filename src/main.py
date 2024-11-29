import time

from fastapi import FastAPI

from src.my_model import MyModel
# from my_model import MyModel 
from pydantic import BaseModel
app = FastAPI()


@app.get("/")
def read_root():
    return {
        "hello":"world"
    }


my_model = MyModel()


@app.get("/v1/get_emotions/")
def text_to_emotions(sentence: str):
    max_label: int = 1
    time_start = time.time()
    predictions = my_model.predicts(sentence)
    time_consumed = time.time() - time_start

    predictions = sorted(predictions, key=lambda tp: tp[1], reverse=True)[:max_label]

    result = []
    for label, probability in predictions:
        label_prob = {
            'label': label,
            'probability': probability
        }
        result.append(label_prob)
    return {
        "time": time_consumed,
        "result": result
    }




# @app.get("/v1/get_emotions/")
# def text_to_emotions(sentence: str, max_label: int = 2):

#     time_start = time.time()
#     predictions = my_model.predicts(sentence)
#     time_consumed = time.time() - time_start

#     predictions = sorted(predictions, key=lambda tp: tp[1], reverse=True)[:max_label]

#     result = []
#     for label, probability in predictions:
#         label_prob = {
#             'label': label,
#             'probability': probability
#         }
#         result.append(label_prob)
#     return {
#         "time": time_consumed,
#         "result": result
#     }



class EmotionRequest(BaseModel):
    sentence: str


