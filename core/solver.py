
class Solver:

    def __init__(self):
        self.iter_ = 0
        self.current_step_ = 0;
        self.net_ = None
        self.test_net_ = None
        self.test_interval_ = 100
        self.test_count_ = 100

    def AddTrainNet(self, net):
        self.net_ = net

    def AddTestNet(self, net):
        self.test_net_ = net

    def Solve(self, iters):
        self.Step(iters)

    def Step(self, iters):
        start_iter = self.iter_
        stop_iter  = self.iter_ + iters

        while self.iter_ < stop_iter:
            if self.iter_ % self.test_interval_ == 0:
                self.Test()
            self.net_.ClearParamDiffs()
            self.net_.ForwardBackward()

            self.iter_ += 1

            self.ApplyUpdate()

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
        count = 0
        total = 0
        for i in range(self.test_count_):
            self.test_net_.Forward()
            count = count + self.test_net_.output_blobs()[0].data()
            total = total + self.test_net_.output_blobs()[1].data()
   
        print 'Accuracy:',count*1.0/total*100,'%'
 
