from MI.G.qsa import q_f


data = pd.read_csv("./wiscon.csv")
qubo = pd.read_csv("./wiscon_qubo.csv")
q_f(data, qubo)