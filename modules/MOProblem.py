from amplify import VariableGenerator
from amplify import solve
from amplify import FixstarsClient
from amplify import Model
import numpy as np
import pandas as pd

api_key = None

def set_api_key(key):
    global api_key
    api_key = key

def get_client(timeout=5000):
    client = FixstarsClient()
    client.token = api_key
    client.parameters.timeout = timeout
    return client

def get_model(data, goal):
    num_recipe = len(data)  # レシピ数
    num_nut = len(goal)  # 栄養素の種類数
    num_type = len(data['データ区分'].unique())

    df = data[data.columns[1:].tolist() + [data.columns[0]]]
    df = df.iloc[:,-num_nut - 1:]
    for i, v in enumerate(goal):
        df.iloc[:,i] /= v

    N = np.array(df.iloc[:, range(num_nut)])
    lam = [1.0] * num_nut  # 簡単のため重要度λαは全て1.0で固定
    q1 = N @ np.diag(lam) @ N.T  # 式(2')の1項目
    for i in range(num_recipe):
        nut_sum = 0
        for alpha in range(num_nut):
            nut_sum += 2 * lam[alpha] * N[i,alpha]  # 目標値tαは1.0に規格化済みのため略
        q1[i,i] -= nut_sum  # 式(2')の2項目

    p = pd.get_dummies(df['データ区分'], dtype=float).values
    q2 = p @ p.T
    q2 -= np.identity(len(q2))  # 対角成分を 0 にする

    gen = VariableGenerator()
    x = gen.array("Binary", num_recipe)

    q3 = q1 + q2
    f = sum(x[i] * q3[i, j] * x[j] for i in range(num_recipe) for j in range(num_recipe))
    model = Model(f)

    return x, model

def find_best_menu(data, goal):
    x, model = get_model(data, goal)

    # Amplify　ソルバ取得，求解
    client = get_client()
    result = solve(model, client)

    if len(result.solutions) == 0:
        raise RuntimeError("Some of the constraints are not satisfied.")

    x_values = result.solutions[0].values
    solution = x.decode(x_values)

    return solution