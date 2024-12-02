import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
import re
import pandas as pd
import numpy as np
from io import StringIO
import csv
from fastapi import File, UploadFile, HTTPException


app = FastAPI()


@app.get("/")
def start_page():
    return 'All Good. App is running.'

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

nm_torque_regex = r'[\d\.]+\s*Nm'
kgm_torque_regex = r'[\d\.]+\s*[@|kgm]'
number_regex = r'[\d\.]+'

def parse_torque(torque_data):
    torque = []
    max_torque_rpm = []
    for item in torque_data:
        if item == '':
            torque.append(None)
            max_torque_rpm.append(None)
            continue

        #torque parsing
        t = re.findall(nm_torque_regex, item, re.IGNORECASE)
        if len(t) == 1:
            torque.append(float(re.findall(number_regex, t[0])[0]))
        else:
            t = re.findall(kgm_torque_regex, item, re.IGNORECASE)
            if len(t) < 1: # then it's "210 / 1900"
                torque.append(float(re.findall(number_regex, item)[0]))
            else:
                torque.append(10 * float(re.findall(number_regex, t[0])[0])) #1kgm = 10Nm

        #max_rpm_parsing
        numbers = re.findall(number_regex, item)
        if len(numbers) < 2: #no rpm
            max_rpm = None
        elif len(numbers) == 2:
            max_rpm = numbers[1]
            if '.' in max_rpm:
                max_rpm = float(max_rpm) * 1000 # dot is used to separate thousands, float(4.500) = 4.5
            else:
                max_rpm = float(max_rpm)
        else:
            if '+/-' in item: #4000+/-500 rpm
                max_rpm = float(numbers[1]) + float(numbers[2])
            else: # 250Nm@ 1500-3000rpm
                max_rpm = float(numbers[2])

        max_torque_rpm.append(max_rpm)


    return torque, max_torque_rpm


def prepare_data(df):
    df['mileage'] = df['mileage'].str.replace('kmpl', '').replace('km/kg', '')
    df['engine'] = df['engine'].str.replace('CC', '')
    df['max_power'] = df['max_power'].str.replace('bhp', '')

    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df['engine'] = pd.to_numeric(df['engine'], errors='coerce')
    df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')

    df['torque'] = df['torque'].str.replace(',', '.')

    df['torque'].fillna(value='', inplace=True)

    torque, max_rpm = parse_torque(df['torque'])

    df['torque'] = torque
    df['max_torque_rpm'] = max_rpm

    with open('median_values.pickle', 'rb') as f:
        train_med = pickle.load(f)

    df.fillna(value=train_med, inplace=True)

    with open('encoder.pickle', 'rb') as f:
        ohe_enc = pickle.load(f)

    X_cat = df[['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']]
    X_cat = ohe_enc.transform(X_cat)

    with open('poly.pickle', 'rb') as f:
        poly = pickle.load(f)

    non_encoded_columns = list(set(df.columns) - set(['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']))
    non_encoded_df = df[non_encoded_columns]

    non_encoded_df = poly.fit_transform(non_encoded_df) 

    data = np.concatenate([X_cat.toarray(), non_encoded_df], axis=1)

    return data


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item_df = pd.DataFrame([json.loads(item.model_dump_json())])

    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

    item_data = prepare_data(item_df)
    return np.exp(model.predict(item_data))


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()

    item_df = pd.read_csv(file.filename)
    
    item_data = prepare_data(item_df)

    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

    return list(np.exp(model.predict(item_data)))
