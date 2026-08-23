import numpy as np
import math
from scipy import special

class ADVar(object):
    def __init__(self, val=0.0, deriv=None, idx=-1, size=None):
        """
        val: 數值
        deriv: 直接傳入 numpy array 作為導數向量
        idx: 若為自變數，其在向量中的索引
        size: 向量的總長度 (N_local)
        """
        self.val = float(val)
        
        if deriv is not None:
            self.deriv = deriv
        elif size is not None:
            self.deriv = np.zeros(size)
            if idx >= 0:
                self.deriv[idx] = 1.0
        else:
            # 預設為純量模式（無導數）
            self.deriv = np.array([])

    # --- 基本屬性與轉換 ---
    def __str__(self):
        return f"value: {self.val} deriv (dense): {self.deriv}"

    def __repr__(self):
        return self.__str__()

    def __float__(self): return float(self.val)
    def __int__(self): return int(self.val)
    def __long__(self): return int(self.val) # Python 3 中 long 與 int 已合併

    def getVal(self): return self.val
    def setVal(self, v): self.val = v

    def getDeriv(self, idx=-1):
        if idx < 0: return self.deriv
        return self.deriv[idx] if idx < len(self.deriv) else 0.0

    # --- 比較函式 ---
    def __lt__(self, other):
        val_other = other.val if isinstance(other, ADVar) else other
        return self.val < val_other

    def __eq__(self, other):
        val_other = other.val if isinstance(other, ADVar) else other
        return self.val == val_other

    def derivEq(self, other):
        if not self.val == other.val: return False
        return np.array_equal(self.deriv, other.deriv)

    def derivApproxEq(self, other, tol=1e-6):
        abs_tol = max(abs(self.val), abs(other.val)) * tol
        if abs(self.val - other.val) >= abs_tol: return False
        return np.allclose(self.deriv, other.deriv, atol=abs_tol)

    # --- 算術運算 (重寫所有魔法方法) ---
    def __add__(self, other):
        if isinstance(other, ADVar):
            return ADVar(self.val + other.val, self.deriv + other.deriv)
        return ADVar(self.val + other, self.deriv.copy())

    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ADVar):
            return ADVar(self.val - other.val, self.deriv - other.deriv)
        return ADVar(self.val - other, self.deriv.copy())

    def __rsub__(self, other):
        if isinstance(other, ADVar):
            return ADVar(other.val - self.val, other.deriv - self.deriv)
        return ADVar(other - self.val, -self.deriv)

    def __mul__(self, other):
        if isinstance(other, ADVar):
            # 乘法法則: d(ab) = b*da + a*db
            return ADVar(self.val * other.val, other.val * self.deriv + self.val * other.deriv)
        return ADVar(self.val * other, self.deriv * other)

    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ADVar):
            # 除法法則: d(a/b) = (b*da - a*db) / b^2
            new_val = self.val / other.val
            new_deriv = (other.val * self.deriv - self.val * other.deriv) / (other.val**2)
            return ADVar(new_val, new_deriv)
        return ADVar(self.val / other, self.deriv / other)

    def __rtruediv__(self, other):
        # other / self -> d(c/u) = -c/u^2 * du
        new_val = other / self.val
        new_deriv = (-other / (self.val**2)) * self.deriv
        return ADVar(new_val, new_deriv)

    def __pow__(self, p):
        # 這裡只實作 self^constant，若要 ADVar^ADVar 則需 log 輔助
        new_val = pow(self.val, p)
        new_deriv = (p * pow(self.val, p - 1)) * self.deriv
        return ADVar(new_val, new_deriv)

    def __abs__(self):
        new_val = abs(self.val)
        sign = np.sign(self.val) if self.val != 0 else 0
        return ADVar(new_val, sign * self.deriv)

    def __pos__(self): return ADVar(self.val, self.deriv.copy())
    def __neg__(self): return ADVar(-self.val, -self.deriv)

# --- 數學函式 (外部定義) ---
def exp(x):
    val = np.exp(x.val)
    return ADVar(val, val * x.deriv)

def log(x):
    return ADVar(np.log(x.val), (1.0 / x.val) * x.deriv)

def sin(x):
    return ADVar(np.sin(x.val), np.cos(x.val) * x.deriv)

def cos(x):
    return ADVar(np.cos(x.val), -np.sin(x.val) * x.deriv)

def sqrt(x):
    val = np.sqrt(x.val)
    return ADVar(val, (0.5 / val) * x.deriv)

def erf(x):
    val = special.erf(x.val)
    deriv = (2.0 / np.sqrt(np.pi) * np.exp(-x.val**2)) * x.deriv
    return ADVar(val, deriv)

def erfc(x):
    val = special.erfc(x.val)
    deriv = (-2.0 / np.sqrt(np.pi) * np.exp(-x.val**2)) * x.deriv
    return ADVar(val, deriv)

# --- 物理模擬專用輔助函式 ---
def aux1(x):
    """ x/sinh(x) """
    y = x.val
    # 數值穩定性處理 (與你原本代碼邏輯一致)
    if abs(y) < 1e-3:
        z = 1.0 - y**2 / 6.0
        pd = -y / 3.0
    else:
        z = y / np.sinh(y)
        pd = (np.sinh(y) - y * np.cosh(y)) / (np.sinh(y)**2)
    return ADVar(z, pd * x.deriv)

def aux2(x):
    """ 1/(1+exp(x)) """
    y = x.val
    z = 1.0 / (1.0 + np.exp(y))
    pd = -np.exp(y) / ((1.0 + np.exp(y))**2)
    return ADVar(z, pd * x.deriv)

def mapADVar(advar, new_size, mapping_dict):
    """ 
    在 Dense 模式中，mapping 意味著將舊向量的元素搬移到新向量的指定位置
    """
    new_deriv = np.zeros(new_size)
    for old_idx, new_idx in mapping_dict.items():
        new_deriv[new_idx] = advar.deriv[old_idx]
    return ADVar(advar.val, new_deriv)
