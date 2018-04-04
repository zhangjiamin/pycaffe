from layer import Layer

class LossLayer(Layer):

    def __init__(self):
        Layer.__init__(self)

    def LayerSetup(self, bottom, top):
        pass

    def Reshape(self, bottom, top):
        self.shape_ = 0
        top[0].Reshape(0)

    def ExactNumBottomBlobs(self):
        return 2

    def AutoTopBlobs(self):
        return True

    def ExactNumTopBlobs(self):
        return 1

    def AllowForceBackward(self, bottom_index):
        return (bottom_index != 1)


