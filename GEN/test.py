import pandas as pd

from q import q_f


data = pd.read_csv("./datasets/wiscon.csv")
qubo = pd.read_csv("./datasets/wiscon_qubo.csv")
q_f(data, qubo)