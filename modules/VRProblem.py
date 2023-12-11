from amplify import BinarySymbolGenerator
from amplify.constraint import one_hot, less_equal
from amplify import einsum
from amplify import sum_poly
from amplify import InequalityFormulation
from amplify import Solver
from amplify.client import FixstarsClient
import numpy as np

api_key = None

def set_api_key(key):
    global api_key
    api_key = key

# クライアントの設定
def create_configured_client():
    # クライアントの設定
    client = FixstarsClient()
    client.token = api_key
    client.parameters.timeout = 5000  # タイムアウト5秒

    return client

def gen_distance_matrix(lat_lon):
    # 距離行列
    all_diffs = np.expand_dims(lat_lon, axis=1) - np.expand_dims(lat_lon, axis=0)
    distances = np.sqrt(np.sum(all_diffs**2, axis=-1))

    return distances

def upperbound_of_tour(capacity: int, demand: np.ndarray) -> int:
    max_tourable_cities = 1
    for w in sorted(demand):
        capacity -= w
        if capacity >= 0:
            max_tourable_cities += 1
        else:
            return max_tourable_cities
    return max_tourable_cities

def make_route_constraints(x):
    x[0][1:][:] = 0
    x[0][0][:] = 1

    x[-1][1:][:] = 0
    x[-1][0][:] = 1

    return x

def make_onetrip_constraints(x) -> list:
    max_tourable_cities = x.shape[0]
    dimension = x.shape[1]
    nvehicle = x.shape[2]

    constraints = [
        one_hot(sum_poly(dimension, lambda j: x[i][j][k]))
        for i in range(max_tourable_cities)
        for k in range(nvehicle)
    ]
    return constraints

def make_onevisit_constraints(x) -> list:
    max_tourable_cities = x.shape[0]
    dimension = x.shape[1]
    nvehicle = x.shape[2]

    constraints = [
        one_hot(
            sum_poly(
                max_tourable_cities, lambda i: sum_poly(nvehicle, lambda k: x[i][j][k])
            )
        )
        for j in range(1, dimension)
    ]
    return constraints

def make_capacity_constraints(x, demand: np.ndarray, capacity: int) -> list:
    max_tourable_cities = x.shape[0]
    dimension = x.shape[1]
    nvehicle = x.shape[2]

    constraints = [
        less_equal(
            demand * x[:, :, k],
            capacity,
            method=InequalityFormulation.RelaxationQuadra,
        )
        / capacity
        / capacity
        for k in range(nvehicle)
    ]

    return constraints

def make_opt_model(capacity, demand, dimension, nvehicle, distance_matrix):
    gen = BinarySymbolGenerator()

    # 積載可能量から1台の車両が訪問できる都市の最大数を計算
    max_tourable_cities = upperbound_of_tour(capacity, demand)

    # 決定変数の配列を生成
    x = gen.array(max_tourable_cities, dimension, nvehicle)

    # ルートの制約を適用
    x = make_route_constraints(x)
    max_tourable_cities = x.shape[0]
    dimension = x.shape[1]
    nvehicle = x.shape[2]

    # 経路の総距離の計算
    xr = x.roll(-1, axis=0)
    objective = einsum("pq,ipk,iqk", distance_matrix, x, xr)

    # 追加の制約
    constraints1 = make_onetrip_constraints(x)
    constraints2 = make_onevisit_constraints(x)
    constraints3 = make_capacity_constraints(x, demand, capacity)
    constraints = sum(constraints1) + sum(constraints2) + sum(constraints3)

    # 最終的なモデルの生成
    model = objective + constraints * np.max(distance_matrix)

    return x, model

def onehot2_sequence(x_values) -> dict[int, list]:
    nvehicle = x_values.shape[2]
    sequence = dict()
    for k in range(nvehicle):
        sequence[k] = np.where(x_values[:, :, k])[1]
    return sequence

def process_sequence(sequence: dict[int, list]) -> dict[int, list]:
    new_seq = dict()
    for k, v in sequence.items():
        v = np.append(v, v[0])
        mask = np.concatenate(([True], np.diff(v) != 0))
        new_seq[k] = v[mask]
    return new_seq

def find_best_tour(data, nvehicle):
    client = create_configured_client()
    ncity = len(data)
    dimension = ncity

    distance_matrix = gen_distance_matrix(data)

    # 各都市における配送需要（重量）を乱数で決定
    # 1固定
    # demand = np.random.randint(50, 100, size=ncity)
    demand = np.array([1] * ncity)
    demand[0] = 0

    capacity = int((sum(demand) / nvehicle) * 1.2)

    x, model = make_opt_model(capacity, demand, dimension, nvehicle, distance_matrix)
    solver = Solver(client)
    result = solver.solve(model)

    if len(result.solutions) == 0:
        raise RuntimeError("Some of the constraints are not satisfied.")

    x_values = result.solutions[0].values

    solution = x.decode(x_values)
    sequence = onehot2_sequence(solution)
    best_tour = process_sequence(sequence)

    return best_tour