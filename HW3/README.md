# HW3：Multi-Armed Bandit 策略實驗報告

作者：張朝翔  
學號：7113056035  

## 📘 作業簡介
本作業探討四種常見的多臂老虎機（Multi-Armed Bandit, MAB）策略，並比較其在靜態環境下的表現：

- Epsilon-Greedy
- Upper Confidence Bound (UCB)
- Softmax
- Thompson Sampling

每個策略皆包含：
- 數學公式（LaTeX 格式）
- ChatGPT 提示語
- Python 實作程式碼
- 模擬圖表（回報、探索比例、臂選擇分布）
- 結果分析與策略比較

---

## 📁 專案結構
HW3/
├── report.tex             # LaTeX 原始報告檔
├── report.pdf             # 編譯後報告 PDF（最終提交版本）
├── main.py                # Python 程式碼：模擬四種 MAB 策略
├── plots/                 # 所有策略圖表結果
│   ├── epsilon_greedy_reward.png
│   ├── epsilon_greedy_explore.png
│   ├── epsilon_greedy_armcount.png
│   ├── ucb_reward.png
│   ├── ucb_explore.png
│   ├── ucb_armcount.png
│   ├── softmax_reward.png
│   ├── softmax_explore.png
│   ├── softmax_armcount.png
│   ├── thompson_reward.png
│   ├── thompson_explore.png
│   ├── thompson_armcount.png
│   └── mab_comparison.png
└── README.md              # 本說明文件

---

## 🧪 環境與參數設定

- 臂數量：10
- 模擬步數：1000
- Reward 分布：$\mathcal{N}(\mu, 1.0)$
- 真實臂報酬：臂 0~9 對應固定值（1.0 至 -0.8）

策略參數如下：
- Epsilon-Greedy：$\epsilon = 0.1$
- UCB：$c = 2.0$
- Softmax：$\tau = 0.5$
- Thompson Sampling：使用 $\mathcal{N}(Q(a), 1/\sqrt{N(a)+\varepsilon})$

---

## 📊 策略觀察與結果

- **Thompson Sampling**：表現最佳，能快速聚焦最優臂
- **Epsilon-Greedy**：整體表現次佳，學習穩定，簡單實用
- **UCB**：初期探索開銷高，導致總體回報略低於 Epsilon-Greedy
- **Softmax**：策略保守穩健，但在固定 reward 設計中效率最低

---

## 📎 備註

本專案為深度強化學習 HW3 作業，圖表由 `main.py` 自動產出，報告採用 XeLaTeX 撰寫，圖文完整，已經過人工審閱與優化，可直接編譯使用。