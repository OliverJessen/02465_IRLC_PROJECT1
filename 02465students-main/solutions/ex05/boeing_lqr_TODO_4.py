    B_discrete = scipy.linalg.inv(model.A) @ (A_discrete - np.eye(model.A.shape[0])) @ model.B 
    d_discrete = scipy.linalg.inv(model.A) @ (A_discrete - np.eye(model.A.shape[0])) @  d 