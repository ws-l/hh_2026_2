#pip install fastapi[standard]
#fastapi dev 파일.py
from fastapi import FastAPI
import pandas as pd
import numpy as np

app = FastAPI()

df = pd.read_csv("api_data.csv")
df.drop(columns = "passorfail", inplace=True)

@app.get("/data")
def date_gen(row: int = 10):
    x = df.sample(row)
    return  x.to_dict(orient="records")
