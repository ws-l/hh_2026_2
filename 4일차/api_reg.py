#pip install fastapi[standard]
#fastapi dev 파일.py
from fastapi import FastAPI
import pandas as pd
import numpy as np

app = FastAPI()

df = pd.read_csv("regression/machineCPU.csv")
df.drop(columns = "y", inplace=True)

@app.get("/data_num")
def date_gen():
    x = df.sample(1)
    return  x.to_dict(orient="records")
