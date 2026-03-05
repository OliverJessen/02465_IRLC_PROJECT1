    def sym_f(self, x, u, t=None): 
        return [x[1], sym.cos(x[0] + u[0])] 