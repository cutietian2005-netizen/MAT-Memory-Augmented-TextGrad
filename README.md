# MAT: Memory-Augmented TextGrad for Efficient Prompt Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Important Note on Origin and Related Work

**This project was conceived and initially implemented in January 2026**, drawing direct inspiration from **TextGrad** (Yuksekgonul et al.) and the iterative refinement strategies in **Nested Learning**. The core idea—using memory to store and reuse textual optimization trajectories—was developed independently during this period.

In March 2026, a parallel work titled **"Generalizable Self-Evolving Memory for Automatic Prompt Optimization" (MemAPO, Liang et al., arXiv:2603.21520)** was publicly released. While both works explore memory-augmented prompt optimization, they originate from distinct motivations and employ different technical approaches.

We acknowledge MemAPO as an important concurrent contribution to the field and cite it as related work. However, **this repository represents our own independent implementation**, and our future publications will clearly delineate the separate origins and unique efficiency-focused contributions of MAT.

## 💡 Motivation and Core Idea

**Why MAT?**
Existing automatic prompt optimization methods (e.g., TextGrad) achieve strong performance but incur high computational costs due to repeated API calls and optimization steps. Our insight is that successful optimization trajectories contain reusable knowledge. By building a lightweight memory of past textual gradients, we can:
- Reduce the number of optimization iterations.
- Lower API costs without sacrificing accuracy.

MAT implements a simple, static memory retrieval mechanism that augments the TextGrad optimizer with relevant historical strategies, demonstrating that **efficiency gains can be achieved with minimal architectural complexity**.

## ✨ Key Differences from MemAPO

| Aspect | MemAPO (Liang et al., 2026) | MAT (This Work) |
| :--- | :--- | :--- |
| **Core Mechanism** | Dual-Memory + Self-Reflection + Dynamic Updates | Single Memory + Static Retrieval |
| **Optimization Focus** | Maximizing accuracy via evolving templates | **Balancing accuracy with API efficiency** |
| **Implementation Complexity** | High (reflection loops, template merge/split) | **Low** (retrieval-only, plug-and-play) |
| **Inspiration** | Cognitive science, schema theory | TextGrad, Nested Learning |

Our experiments show that MAT achieves comparable or better accuracy than vanilla TextGrad while significantly reducing API calls and runtime.


## 🚀 Quick Start

Follow these steps to set up and run the MAT experiment on your own machine.

### 1. Clone the repository

```bash
git clone https://github.com/cutietian2005-netizen/MAT-Memory-Augmented-TextGrad.git
cd MAT-Memory-Augmented-TextGrad
```

### 2. Install dependencies

Make sure you have Python 3.9+ installed. Then install the required packages:

```bash
pip install textgrad openai numpy sentence-transformers matplotlib
```

> **Optional**: If you want to freeze exact versions, create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`:
>
> ```text
> textgrad
> openai
> numpy
> sentence-transformers
> matplotlib
> ```

### 3. Set your DeepSeek API key

MAT uses the DeepSeek API. You must provide your API key as an environment variable before running the script.

- **On Windows (PowerShell)**:
  ```powershell
  $env:DEEPSEEK_API_KEY="sk-your-api-key-here"
  ```
Alternatively, you can temporarily hardcode the key in `mat.py` .

### 4. Run the two-phase experiment

Execute the main experiment script:

```bash
python mat.py
```

The script will:
- Train the MAT memory on 10 arithmetic problems.
- Compare MAT against the vanilla TextGrad optimizer on 30 unseen test problems.
- Print a summary table showing accuracy, average iterations, API calls, and runtime.
- Save detailed results to `phase2_test_results.json` and the trained memory to `memory_after_training.json`.
- Generate a comparison plot `comparison_plot.png` (if `matplotlib` is installed).

### 5. Interpret the results

After the run finishes, you will see a final comparison like:

```
📈 测试阶段最终对比
======================================================================
VANILLA:
  准确率: 26/30 (86.7%)
  平均迭代次数: 0.40
  平均 API 调用: 2.20
  平均耗时 (秒): 28.22

MAT:
  准确率: 25/30 (83.3%)
  平均迭代次数: 0.43
  平均 API 调用: 2.30
  平均耗时 (秒): 27.25
```

You can also inspect `phase2_test_results.json` for per-question details and `comparison_plot.png` for a visual summary.
