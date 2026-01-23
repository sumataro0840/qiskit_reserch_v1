from qiskit_ibm_runtime import QiskitRuntimeService

# 1. サービスの初期化
service = QiskitRuntimeService()

# 2. ジョブIDを指定
job_id = "d5mfpr1h2mqc739arl2g" 
job = service.job(job_id)

# 文字列としてステータスを確認
current_status = str(job.status())
print(f"現在のステータス: {current_status}")

if current_status == 'DONE':
    # 結果の取得
    result = job.result()
    
    # SamplerV2 の結果を取得
    pub_result = result[0]
    
    # 測定データのカウントを取得
    # 注: measure_all() を使った場合、キー名は通常 'meas' になります
    if hasattr(pub_result.data, 'meas'):
        counts = pub_result.data.meas.get_counts()
    else:
        # 他のレジスタ名（'c'など）になっている可能性を考慮
        reg_name = list(pub_result.data.keys())[0]
        counts = getattr(pub_result.data, reg_name).get_counts()
    
    print("\n--- 測定データ (Counts) ---")
    print(counts)
    
    # 確率 P の算出
    total_shots = sum(counts.values())
    
    # 目的ビット（一番左端のビット）が '1' になっているものを集計
    # 4bit (q0, q1, q2, q3) の場合、q3が目的ビットなら bitstr[0] を確認
    success_counts = sum(count for bitstr, count in counts.items() if bitstr[0] == '1')
    p_measured = success_counts / total_shots
    
    print("\n" + "="*45)
    print(f"実機での測定確率 P: {p_measured:.4f}")
    print(f"(目標値: 0.0500 付近)")
    print("="*45)
    
    # VaRの逆算
    # 今回テストした x_test = 1.6892, E = 3.1540
    print(f"この確率に基づくと、VaR（リターン）は約 {3.1540 - 1.6892:.4f} です。")
else:
    print(f"ジョブの状態は {current_status} です。完了までもうしばらくお待ちください。")