import csv
from collections import Counter
from collections import namedtuple
from pprint import pprint
from typing import List
from collections import defaultdict


DATA_FILE = "data/farm_prices.csv"
Farm = namedtuple("Farm", ["mh", "price", "coef"])

def calculate_coefs() -> List[Farm]:
    data = []
    with open(DATA_FILE) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue

            mh = int(row[0])
            price = int(row[1])
            coef = price / mh
            farm = Farm(mh=mh, price=price, coef=coef)
            data.append(farm)
    return data


def price_clusters(data):
    """ price per 1 MH """
    clusters = defaultdict(lambda: 0)
    step = 50
    rounding = 0
    for farm in data:
        cut = farm.coef // step
        key = f"{round(cut*step, rounding)} - {round((cut+1)*step, rounding)}"
        clusters[key] += 1
    return clusters


def mh_clusters(data):
    """ price per 1 MH """
    data.sort(key=lambda x: x.mh)
    clusters = defaultdict(list)
    step = 25
    rounding = 0
    for farm in data:
        cut = farm.mh // step
        key = f"{round(cut*step, rounding)} - {round((cut+1)*step, rounding)}"
        clusters[key].append(farm.coef)

    for k, v in clusters.items():
        clusters[k] = round(sum(v) / len(v), 1)
    return clusters

data = calculate_coefs()

print("Total farms:", len(data))
print()
print('Price clusters:')
pprint(price_clusters(data))
print()
# '400.0 - 500.0': 1,
# '500.0 - 600.0': 10,
# '600.0 - 700.0': 32,
# '700.0 - 800.0': 8,
# '800.0 - 900.0': 4,
# '900.0 - 1000.0': 2
print('MH clusters:')
pprint(mh_clusters(data))
print()


OUR_FARM_MH = 134
price_per_mh = 633.5
price = OUR_FARM_MH * price_per_mh
uah_usd = 0.036914

print("OUR_FARM_MH:", OUR_FARM_MH)
print("price_per_mh:", price_per_mh)
print("Our price (UAH):", price)
print("Our price (USD):", round(price * uah_usd, 2))



# bot_dau = 1500
# print(f"{bot_dau=}")
# print("Our profit:")
# for i in range(1, 11):
#     k = round(i*0.1, 1)
#     print(f"{k}: {round(k * bot_dau, 1)}")
