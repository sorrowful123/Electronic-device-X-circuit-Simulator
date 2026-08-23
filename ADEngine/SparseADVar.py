


import math
import numpy as np
from scipy import special

class SparseADVar(object):
    def __init__(self, val=0.0, deriv=None, idx=-1):
        self.val = float(val)
        # 使用 dictionary 存儲 {index: derivative}
        if deriv is not None:
            self.deriv = deriv
        elif idx >= 0:
            self.deriv = {idx: 1.0}
        else:
            self.deriv = {}

    # --- 內部輔助函式 ---
    def _calcDeriv(self, other, func):
        """
        模擬原本的 _calcDeriv，但在 dictionary 下更簡單。
        func: lambda dx, dy -> new_dx
        """
        new_deriv = {}
        # 獲取所有出現過的索引 (聯集)
        all_indices = set(self.deriv.keys()) | set(other.deriv.keys())
        for i in all_indices:
            dx = self.deriv.get(i, 0.0)
            dy = other.deriv.get(i, 0.0)
            res = func(dx, dy)
            if res != 0:
                new_deriv[i] = res
        return new_deriv

    # --- 基本轉換與屬性 ---
    def __str__(self):
        return f"value: {self.val} deriv (sparse): {self.deriv}"

    def __repr__(self):
        return self.__str__()

    def __float__(self): return float(self.val)
    def __int__(self): return int(self.val)
    def __long__(self): return int(self.val)

    def getVal(self): return self.val
    def setVal(self, v): self.val = v

    def getDeriv(self, idx=-1):
        if idx < 0: return self.deriv
        return self.deriv.get(idx, 0.0)

    # --- 比較運算 ---
    def __lt__(self, other):
        other_val = other.val if isinstance(other, SparseADVar) else other
        return self.val < other_val

    def __eq__(self, other):
        other_val = other.val if isinstance(other, SparseADVar) else other
        return self.val == other_val

    def derivEq(self, other):
        if self.val != other.val: return False
        return self.deriv == other.deriv

    def derivApproxEq(self, other, tol=1e-6):
        abs_tol = max(abs(self.val), abs(other.val)) * tol
        if abs(self.val - other.val) >= abs_tol: return False
        
        all_indices = set(self.deriv.keys()) | set(other.deriv.keys())
        for i in all_indices:
            if abs(self.deriv.get(i, 0.0) - other.deriv.get(i, 0.0)) >= abs_tol:
                return False
        return True

    # --- 四則運算 ---
    def __add__(self, other):
        if not isinstance(other, SparseADVar):
            return SparseADVar(self.val + other, self.deriv.copy())
        return SparseADVar(self.val + other.val, self._calcDeriv(other, lambda dx, dy: dx + dy))

    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, SparseADVar):
            return SparseADVar(self.val - other, self.deriv.copy())
        return SparseADVar(self.val - other.val, self._calcDeriv(other, lambda dx, dy: dx - dy))

    def __rsub__(self, other):
        if not isinstance(other, SparseADVar):
            new_deriv = {i: -dx for i, dx in self.deriv.items()}
            return SparseADVar(other - self.val, new_deriv)
        return other.__sub__(self)

    def __mul__(self, other):
        if not isinstance(other, SparseADVar):
            return SparseADVar(self.val * other, {i: dx * other for i, dx in self.deriv.items()})
        # d(ab) = b*da + a*db
        return SparseADVar(self.val * other.val, self._calcDeriv(other, lambda dx, dy: other.val * dx + self.val * dy))

    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, SparseADVar):
            return SparseADVar(self.val / other, {i: dx / other for i, dx in self.deriv.items()})
        # d(a/b) = (b*da - a*db) / b^2
        new_val = self.val / other.val
        inv_b2 = 1.0 / (other.val**2)
        return SparseADVar(new_val, self._calcDeriv(other, lambda dx, dy: (other.val * dx - self.val * dy) * inv_b2))

    def __rtruediv__(self, other):
        # other / self -> -other/self^2 * d(self)
        new_val = other / self.val
        coeff = -other / (self.val**2)
        return SparseADVar(new_val, {i: dx * coeff for i, dx in self.deriv.items()})

    def __pow__(self, other):
        if isinstance(other, SparseADVar):
            new_val = pow(self.val, other.val)
            # 廣義冪法則: d(u^v) = u^v * (v/u * du + ln(u) * dv)
            t1 = other.val * pow(self.val, other.val - 1)
            t2 = math.log(self.val) * new_val
            return SparseADVar(new_val, self._calcDeriv(other, lambda dx, dy: t1 * dx + t2 * dy))
        else:
            new_val = pow(self.val, other)
            coeff = other * pow(self.val, other - 1)
            return SparseADVar(new_val, {i: dx * coeff for i, dx in self.deriv.items()})

    def __abs__(self):
        sign = 1.0 if self.val > 0 else (-1.0 if self.val < 0 else 0.0)
        return SparseADVar(abs(self.val), {i: dx * sign for i, dx in self.deriv.items()})

    def __pos__(self): return SparseADVar(self.val, self.deriv.copy())
    def __neg__(self): return SparseADVar(-self.val, {i: -dx for i, dx in self.deriv.items()})

# --- 外部數學函式 ---
def exp(x):
    val = math.exp(x.val)
    return SparseADVar(val, {i: dx * val for i, dx in x.deriv.items()})

def log(x):
    return SparseADVar(math.log(x.val), {i: dx / x.val for i, dx in x.deriv.items()})

def sin(x):
    val = math.sin(x.val)
    cos_val = math.cos(x.val)
    return SparseADVar(val, {i: dx * cos_val for i, dx in x.deriv.items()})

def cos(x):
    val = math.cos(x.val)
    sin_val = math.sin(x.val)
    return SparseADVar(val, {i: -dx * sin_val for i, dx in x.deriv.items()})

def sqrt(x):
    val = math.sqrt(x.val)
    coeff = 0.5 / val
    return SparseADVar(val, {i: dx * coeff for i, dx in x.deriv.items()})

def Pow(x, p):
    # 支援常數底數 x
    if not isinstance(x, SparseADVar) and isinstance(p, SparseADVar):
        val = pow(x, p.val)
        coeff = math.log(x) * val
        return SparseADVar(val, {i: dp * coeff for i, dp in p.deriv.items()})
    return x.__pow__(p)

def aux1(x):
    y = x.val
    if abs(y) < 1e-4: # 泰勒展開處理接近 0 的情況
        z = 1.0 - y**2 / 6.0
        pd = -y / 3.0
    else:
        z = y / math.sinh(y)
        pd = (math.sinh(y) - y * math.cosh(y)) / (math.sinh(y)**2)
    return SparseADVar(z, {i: dx * pd for i, dx in x.deriv.items()})

def aux2(x):
    y = x.val
    z = 1.0 / (1.0 + math.exp(y))
    pd = -math.exp(y) / ((1.0 + math.exp(y))**2)
    return SparseADVar(z, {i: dx * pd for i, dx in x.deriv.items()})

def erf(x):
    val = special.erf(x.val)
    coeff = 2.0 / math.sqrt(math.pi) * math.exp(-x.val**2)
    return SparseADVar(val, {i: dx * coeff for i, dx in x.deriv.items()})

def erfc(x):
    val = special.erfc(x.val)
    coeff = -2.0 / math.sqrt(math.pi) * math.exp(-x.val**2)
    return SparseADVar(val, {i: dx * coeff for i, dx in x.deriv.items()})

def mapADVar(advar, mapping_dict):
    """ mapping_dict: {old_idx: new_idx} """
    new_deriv = {}
    for old_idx, d in advar.deriv.items():
        if old_idx in mapping_dict:
            new_deriv[mapping_dict[old_idx]] = d
    return SparseADVar(advar.val, new_deriv)
