        s = Variable(torch.FloatTensor(s))
        x = self.feature(s)
        advantage = self.advantage(x)
        value = self.value(x) 