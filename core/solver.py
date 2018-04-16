
class Solver:

    def __init__(self):
        self.iter_ = 0
        self.current_step_ = 0;
        self.net_ = None
        self.test_net_ = None

    def AddTrainNet(self, net):
        self.net_ = net

    def AddTestNet(self, net):
        self.test_net_ = net

    def Solve(self):
        pass

    def Step(self, iters):
        start_iter = self.iter_
        stop_iter  = self.iter_ + iters

        while self.iter_ < stop_iter:
            self.net_.ClearParamDiffs()
            self.net_.ForwardBackward()

            self.ApplyUpdate()
            self.iter_ += 1

    def net(self):
        return self.net_

    def test_net(self):
        return self.test_net_

    def iter(self):
        return self.iter_

    def type(self):
        return ''

    def ApplyUpdate(self):
        pass

    def Test(self):
        pass

    
