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

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MAT-prompt-optimization.git
   cd MAT-prompt-optimization
