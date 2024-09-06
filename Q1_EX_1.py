import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from multiprocessing import Pool

# 参数设置
p0 = 0.1  # 可接受质量水平（标称值）
p1 = 0.15  # 拒绝质量水平
alpha = 0.05  # 第一类错误概率
beta = 0.1  # 第二类错误概率

# 计算决策边界
lnA = np.log((1-beta)/alpha)
lnB = np.log(beta/(1-alpha))

# SPRT模拟函数
def sprt_simulation(p0, p1, lnA, lnB, true_p):
    n = 0
    sum_x = 0
    decision = "decision"
    while True:
        n += 1
        x = np.random.rand() < true_p  # 模拟抽样
        sum_x += x
        lnL = sum_x * np.log(p1/p0) + (n-sum_x) * np.log((1-p1)/(1-p0))
        # 做出决策
        if lnL >= lnA:
            decision = "Reject Batch"
            return "Reject Batch"  # 拒绝批次
        elif lnL <= lnB:
            decision = "Reject Batch"
            return "Accept Batch"  # 接受批次
        else:
            continue  # 继续抽样

    return decision, n

# 模拟不同真实次品率下的SPRT性能
true_p_range = np.arange(0.05, 0.26, 0.01)
num_simulations = 10000
oc_curve = np.zeros_like(true_p_range)
asn_curve = np.zeros_like(true_p_range)

def simulate_sprt(true_p):
    decisions = []
    sample_sizes = []
    for _ in range(num_simulations):
        decision, n = sprt_simulation(p0, p1, lnA, lnB, true_p)
        decisions.append(decision)
        sample_sizes.append(n)
    return decisions, sample_sizes

with Pool() as pool:
    results = pool.map(simulate_sprt, true_p_range)

for i, (decisions, sample_sizes) in enumerate(results):
    oc_curve[i] = sum(d == 'accept' for d in decisions) / num_simulations
    asn_curve[i] = np.mean(sample_sizes)

# 绘制OC曲线
plt.figure(figsize=(10, 6))
plt.plot(true_p_range, oc_curve, 'b-', linewidth=2)
plt.axvline(x=p0, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=p1, color='r', linestyle='--', linewidth=1.5)
plt.title('操作特性（OC）曲线', fontsize=14)
plt.xlabel('真实次品率', fontsize=12)
plt.ylabel('接受概率', fontsize=12)
plt.legend(['OC曲线', 'p0', 'p1'], loc='best')
plt.grid(True)
plt.savefig('问题1_OC曲线.png', dpi=300)
plt.close()

# 绘制ASN曲线
plt.figure(figsize=(10, 6))
plt.plot(true_p_range, asn_curve, 'g-', linewidth=2)
plt.axvline(x=p0, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=p1, color='r', linestyle='--', linewidth=1.5)
plt.title('平均样本量（ASN）曲线', fontsize=14)
plt.xlabel('真实次品率', fontsize=12)
plt.ylabel('平均样本量', fontsize=12)
plt.legend(['ASN曲线', 'p0', 'p1'], loc='best')
plt.grid(True)
plt.savefig('问题1_ASN曲线.png', dpi=300)
plt.close()

# 计算在p0和p1处的具体性能指标
idx_p0 = np.argmin(np.abs(true_p_range - p0))
idx_p1 = np.argmin(np.abs(true_p_range - p1))

p0_performance = [oc_curve[idx_p0], asn_curve[idx_p0]]
p1_performance = [oc_curve[idx_p1], asn_curve[idx_p1]]

# 输出结果
print('SPRT模型性能指标：')
print(f'在p0 ({p0}) 处的接受概率: {p0_performance[0]}')
print(f'在p0 ({p0}) 处的平均样本量: {p0_performance[1]}')
print(f'在p1 ({p1}) 处的接受概率: {p1_performance[0]}')
print(f'在p1 ({p1}) 处的平均样本量: {p1_performance[1]}')

# 保存结果到Excel文件
results_df = pd.DataFrame({
    '真实次品率': true_p_range,
    '接受概率': oc_curve,
    '平均样本量': asn_curve
})
results_df.to_excel('问题1_SPRT性能指标.xlsx', index=False)

# 模拟实际检测过程
num_batches = 1000
batch_decisions = []
batch_sample_sizes = []

def simulate_batch():
    true_p = max(0, min(1, 0.10 + 0.05 * np.random.randn()))  # 模拟批次间的质量波动
    decision, n = sprt_simulation(p0, p1, lnA, lnB, true_p)
    return decision, n

with Pool() as pool:
    results = pool.starmap(simulate_batch, [() for _ in range(num_batches)])

batch_decisions, batch_sample_sizes = zip(*results)

# 统计结果
reject_rate = sum(d == 'reject' for d in batch_decisions) / num_batches
avg_sample_size = np.mean(batch_sample_sizes)

# 输出实际检测结果
print('实际检测结果：')
print(f'拒收率: {reject_rate}')
print(f'平均样本量: {avg_sample_size}')

# 绘制样本量分布直方图
plt.figure(figsize=(10, 6))
plt.hist(batch_sample_sizes, bins='auto', density=True)
plt.title('样本量分布', fontsize=14)
plt.xlabel('样本量', fontsize=12)
plt.ylabel('频率', fontsize=12)
plt.grid(True)
plt.savefig('问题1_样本量分布.png', dpi=300)
plt.close()

# 保存实际检测结果到Excel文件
actual_results_df = pd.DataFrame({
    '批次': range(1, num_batches + 1),
    '决策': batch_decisions,
    '样本量': batch_sample_sizes
})
actual_results_df.to_excel('问题1_实际检测结果.xlsx', index=False)

