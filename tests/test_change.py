import numpy as np,torch
from change_lab.models import ChangeNet
from change_lab.experiment import confusion,metrics,infer

def test_shared_siamese_symmetry_and_odd_shape():
    torch.set_num_threads(2);model=ChangeNet(True).eval();x=torch.randn(1,8,17,19)
    torch.testing.assert_close(model(x),model(torch.cat([x[:,4:],x[:,:4]],1)))
    assert model(x).shape==(1,1,17,19)

def test_edge_tiles_and_confusion():
    model=ChangeNet();x=np.zeros((8,19,23),np.float32);p=infer(model,x,tile=16)
    assert p.shape==(19,23) and np.isfinite(p).all()
    assert metrics(confusion([1,0,1,0],[1,1,0,0]))['iou']==1/3
