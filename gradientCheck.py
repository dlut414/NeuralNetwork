import numpy as np;

def gradientCheck(Neurons, predict, J, x, y):
    sigma = 0.0001;
    nLayer = len(Neurons);
    '''
    a = predict(x);
    dPredict(dJ(a, y), x);
    '''
    Dw = [];
    Db = [];
    for i in range(1, nLayer):
        Dw.append(Neurons[i].Dw);
        Db.append(Neurons[i].Db);
    Dw_val = [];
    Db_val = [];
    for i in range(1, nLayer):
        Dw_tmp = Neurons[i].Dw;
        Db_tmp = Neurons[i].Db;
        Dw_v = np.ndarray(Dw_tmp.shape);
        Db_v = np.ndarray(Dw_tmp.shape);
        for r in range(0, Dw_tmp.shape[0]):
            for c in range(0, Dw_tmp.shape[1]):
                Neurons[i].w[r][c] += sigma;
                J1 = J(predict(x), y);
                Neurons[i].w[r][c] -= 2* sigma;
                J2 = J(predict(x), y);
                Dw_v[r][c] = (J1 - J2) / (2* sigma);
                Neurons[i].w[r][c] += sigma;
        for r in range(0, Db_tmp.shape[0]):
            for c in range(0, Db_tmp.shape[1]):
                Neurons[i].b[r][c] += sigma;
                J1 = J(predict(x), y);
                Neurons[i].b[r][c] -= 2* sigma;
                J2 = J(predict(x), y);
                Db_v[r][c] = (J1 - J2) / (2* sigma);
                Neurons[i].b[r][c] += sigma;
        Dw_val.append(Dw_v);
        Db_val.append(Db_v);
        Neurons[i].Dw = Dw_tmp;
        Neurons[i].Db = Db_tmp;
        
    for i in range(0, nLayer-1):
        for r in range(0, Dw[i].shape[0]):
            for c in range(0, Dw[i].shape[1]):
                print " Dw: ", Dw[i][r][c], "; ", " Dw_val: ", Dw_val[i][r][c];
                print " difference: ", Dw[i][r][c] - Dw_val[i][r][c];
        for r in range(0, Db[i].shape[0]):
            for c in range(0, Db[i].shape[1]):
                print " Db: ", Db[i][r][c], "; ", " Db_val: ", Db_val[i][r][c];
                print " difference: ", Db[i][r][c] - Db_val[i][r][c];

    return;
