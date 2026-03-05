    k = 0 
    expected_number_of_items = sum([p * model.f(x, u, w, k=0) for w, p in model.Pw(x, u, k).items()]) 