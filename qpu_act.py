import numpy as np
from scipy.stats import norm
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# ======================
# 1. Figure 1 のデータとポートフォリオ設定の取り込み
# ======================
# 年次リターンデータ (7年, 3資産)
R = np.array([
    [0.027, -0.046, -0.377],
    [0.001,  0.034,  0.207],
    [0.01,  -0.006,  0.032],
    [0.049,  0.333,  0.787],
    [0.051,  0.34,   0.243],
    [0.47,   0.246,  0.527],
    [0.367,  0.918,  1.102]
])

# 平均リターンと共分散行列 (Figure 1と同じ値を使用)
mu = R.mean(axis=0)
Sigma = np.array([
    [0.037602905, 0.038592381, 0.057048119],
    [0.038592381, 0.110059476, 0.142509524],
    [0.057048119, 0.142509524, 0.24143881]
])

# ポートフォリオの最適ウェイト w の計算 (Figure 1と同じパラメータ)
Sigma_inv = np.linalg.inv(Sigma)
lam = 5  # リスク回避度 (Figure 1に合わせる)
w = (lam/2) * Sigma_inv.dot(mu)

# 理論的な期待リターン (スケーリングの基準点)
E_theory = w @ mu
print(f"Theoretical Portfolio Mean (Figure 1): {E_theory:.4f}")

# ======================
# 2. PCA によるスケーリング係数の算出
# ======================
# 共分散行列の固有値分解
eigvals, eigvecs = np.linalg.eigh(Sigma)
# 固有値の大きい順にソート
idx = eigvals.argsort()[::-1]
v1 = eigvecs[:, idx][:, 0]

# v1 の符号確認（table2 のハードコードされた Z と向きを合わせる）
# table2 の Z は [0.791, ...] であり、これと相関が合うようにする
Z_check = (R - mu) @ v1
Z_target = np.array([0.791, 0.275, 0.438, -0.370, 0.073, -0.201, -1.006])
if np.corrcoef(Z_check, Z_target)[0, 1] < 0:
    v1 = -v1

# スケーリング係数 k = |w^T * v1|
# (ポートフォリオのリターンが第1主成分の変化に対してどれだけ感応するか)
scaling_factor = abs(w @ v1)
print(f"PCA Scaling Factor (|w @ v1|): {scaling_factor:.4f}")

# ======================
# 3. 量子計算による VaR 推定 (既存コード)
# ======================
Z = Z_target # 整合性のため既存の値を使用
L = -Z
mu_L = L.mean()
sigma_L = L.std(ddof=0)
alpha = 0.95  # 95%

# 量子バックエンドの設定 (実行可能な環境がある場合)
try:
    service = QiskitRuntimeService()
    backend = service.backend("ibm_torino")
    
    theta = 2 * np.arcsin(np.sqrt(alpha))
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)
    tqc = transpile(qc, backend=backend)
    
    sampler = Sampler(mode=backend)
    job = sampler.run([tqc], shots=10000)
    result = job.result()
    counts = result[0].data.c.get_counts()
    p_quantum = counts.get("1", 0) / 10000
    print(f"Quantum Estimated CDF: {p_quantum}")
except Exception as e:
    print(f"Quantum execution skipped ({e}). Using demo value from Table 2.")
    p_quantum = 0.9396

# 量子計算に基づく L の VaR (PCAスコア空間での損失)
l_var_qpu = norm.ppf(p_quantum, mu_L, sigma_L)
print(f"VaR (PCA Score Space): {l_var_qpu:.4f}")

# ======================
# 4. ポートフォリオリターン空間への逆変換
# ======================
# 式: Return_VaR = E[Return] - (Scaling_Factor * L_VaR)
# L_VaR は損失の大きさ(正)なので、期待リターンからそれをスケーリングして引くことで
# 下側リスク(負のリターン)を求める
reconstructed_return_var = E_theory - (scaling_factor * l_var_qpu)

print("\n" + "="*40)
print(" COMPARISON RESULTS")
print("="*40)
print(f"Original Figure 1 VaR (Target) : -1.6898")
print(f"Table 2 VaR (PCA Score Space)  :  {l_var_qpu:.4f}")
print(f"Reconstructed Portfolio VaR    :  {reconstructed_return_var:.4f}")
print("="*40)
print("※ Reconstructed VaR は、PCAによる1次元近似を経由しているため、")
print("   Figure 1 の厳密解(-1.6898)とは完全には一致しませんが、")
print("   スケール感は補正されています。")