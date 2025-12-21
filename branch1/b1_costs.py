import numpy as np

QQt = np.diag([10.0, 10.0, 1.0, 1.0]) * 1e-1
RRt = 0.01 * np.eye(1)

def stagecost(x: np.ndarray, u:np.ndarray, x_ref: np.ndarray, u_ref: np.ndarray):
    # x should be (4,), u should be (1,)
    x = x[:, None] # make column vector (4,1)
    u = u[:, None] # make column vector (1,1)

    x_ref = x_ref[:, None] # make column vector (4,1)
    u_ref = u_ref[:, None] # make column vector (1,1)

    ll = 0.5 * (x - x_ref).T @ QQt @ (x - x_ref) + 0.5 * (u - u_ref).T @ RRt @ (u - u_ref)

    lx = QQt @ (x - x_ref)
    lu = RRt @ (u - u_ref)

    return ll.squeeze(), lx, lu

def termcost(xT: np.ndarray, xT_ref: np.ndarray, QT: np.ndarray):
    llT = 0.5 * (xT - xT_ref).T @ QT @ (xT - xT_ref)

    lTx = QT @ (xT - xT_ref)

    return llT.squeeze(), lTx