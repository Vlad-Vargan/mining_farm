import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preprocessing
from sklearn.linear_model import LinearRegression
import numpy
from typing import Tuple
import pickle

import datetime as dt

SAVE_FILE = "model.pickle"
FARM_HASHRATE = 100
DAY_ETH = 2 * 6500
ETH_PRICE = 4300


HARD = {
    "1060": 2,
    "P106-100": 3
}

COINS = {
    "ETHTEREUM":{
        "price": 4000,
        "blocks": 11_520,
        "hashrate": 800_000_000_000_000,
        "algo": "Ethash",
        "hard": {"1060": 20_010_000, "P106-100": 21_840_000}
    },
    "ZCASH": {
        "price": 180,
        "blocks": 2_880,
        "hashrate": 8_000_000_000,
        "algo": "Equihash",
        "hard": {"1060": 290, "P106-100": 290}
    },
    "BEAM": {
        "price": 0.8,
        "blocks": 57_600,
        "hashrate": 600_000,
        "algo": "BeamHashIII",
        "hard": {"1060": 9.68, "P106-100": 11.26}
    },
    "RAVENCOIN": {
        "price": 0.09,
        "blocks": 7_200_000,
        "hashrate": 8_000_000_000_000,
        "algo": "X16RV2",
        "hard":  {"1060": 9_290_000, "P106-100": 9_980_000}
    },
    # "ERGO": {
    #     "price": 6.2,
    #     "hashrate": 20_000_000_000_000,
    #     "algo": "Autolykos2",
    #     "hard":  {"1060": 36_370_000, "P106-100": 44_750_000}
    # }
}



def prepare_data() -> Tuple[numpy.ndarray, numpy.ndarray]:
    hashrate_df = pd.read_csv("data/ETH/NetworkHash.csv")

    hashrate_df['timestamp'] = pd.to_datetime(hashrate_df['UnixTimeStamp'], unit='s')
    hashrate_df.set_index("timestamp")

    hashrate_df = hashrate_df.loc[hashrate_df["timestamp"] >= '2020-11-01']

    hashrate_df['timestamp']=hashrate_df['timestamp'].map(dt.datetime.toordinal)
    hashrate_df['Value'] = hashrate_df['Value'].astype(int)

    print(hashrate_df.head(5))

    X = hashrate_df[["timestamp"]].values
    y = hashrate_df[["Value"]].values

    return X, y


def train(X, y) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)

    print(f"{model.coef_=}")
    print(f"{model.intercept_=}")

    return model


def save_model(model: LinearRegression) -> None:
    with open(SAVE_FILE, "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model() -> LinearRegression:
    with open(SAVE_FILE, "rb") as handle:
        unserialized_model = pickle.load(handle)
    return unserialized_model


def _predict_hashrate(model: LinearRegression, year: int, month: int) -> float:
    date = dt.datetime(year=year, month=month, day=1)
    ordinal = dt.datetime.toordinal(date)
    prediction = model.predict([[ordinal]])[0][0]
    return prediction


def predict_profit(model: LinearRegression, year: int, month: int) -> float:
    prediction_Gh = _predict_hashrate(model, year, month)
    hash_part = FARM_HASHRATE / (prediction_Gh*1000)  # convert to Mh
    day_profit = DAY_ETH * ETH_PRICE * hash_part
    month_profit = day_profit * 30
    return month_profit


if __name__ == "__main__":
    # X, y = prepare_data()
    # model = train(X, y)
    # save_model(model)

    # for title, coin in COINS.items():
    #     farm_hash = coin["hard"]["1060"]*2+coin["hard"]["P106-100"]*3
    #     farm_part = farm_hash / coin["hashrate"]
    #     profit = farm_part * coin["blocks"] * coin["price"] * 30
    #     print(title, round(profit, 2), farm_hash)

    model = load_model()

    total_negative = 3735
    total = 0

    m = 1
    print("\nm  date \t prof \t accum \t anti\n")
    for year in (2021, 2022, 2023, 2024):
        for month in range(1, 13):
            if year == 2021 and month < 11:
                continue
            if year == 2024 and month > 3:
                continue
            profit = int(predict_profit(model, year, month))
            total_negative -= profit
            total += profit

            print(f"{m}: {year}/{month}\t", profit, "\t", total, "\t", total_negative)
            m += 1

    # hard = COINS["ETHTEREUM"]["hard"]
    # print((hard["1060"] * 2 + hard["P106-100"] * 3) / 1_000_000 + 28.671) # + 1070


    # plt.scatter(X, y)
    # modeled_rate = model.predict(X)
    # plt.plot(X, modeled_rate, color="r")
    # plt.show()

