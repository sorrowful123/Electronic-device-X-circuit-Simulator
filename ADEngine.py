# ADEngine.py
import SparseADVar as S
import DenseADVar as D

class ADEngine:
    _mode = 'sparse'
    _size = 0

    @staticmethod
    def initialize(mode='sparse', size=0):
        ADEngine._mode = mode
        ADEngine._size = size

    @staticmethod
    def Var(val, idx=-1):
        if ADEngine._mode == 'dense':
            return D.ADVar(val, idx=idx, size=ADEngine._size)
        else:
            return S.SparseADVar(val, idx=idx)

# 使用範例
from ADEngine import ADEngine as AD

# 1. 跑電路模擬器
AD.initialize(mode='sparse')
v_node = AD.Var(5.0, idx=1) 

# 2. 跑 PDE Subdomain
AD.initialize(mode='dense', size=100)
v_flux = AD.Var(0.0, idx=50)
