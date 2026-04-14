"""
Memory-Augmented TextGrad (MAT) - 论文实验最终版（修正版）
修正内容：
1. 恢复旧版正确的梯度注入逻辑（set.add(tg.Variable)），避免覆盖原始梯度。
2. 添加 Hugging Face 镜像支持，解决国内无法下载模型的问题。
3. 将记忆检索相似度阈值降低至 0.4，提高经验召回率。
4. 保留本地 Embedding 方案，并给出切换至 BGE-M3 API 的注释选项。

日期：2026年2月（修正于4月）

使用方法：
1. 设置环境变量 DEEPSEEK_API_KEY（或直接填入下方，提交前删除）
2. pip install textgrad openai numpy sentence-transformers matplotlib
3. python python_20260413_29bfe8_fixed.py

输出：
- phase2_test_results.json  : 详细测试结果
- memory_after_training.json: 训练后的记忆库
- comparison_plot.png       : Vanilla vs MAT 对比图
"""

import os
import json
import time
import re
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple

# ==================== 国内网络加速：Hugging Face 镜像 ====================
# 解决 sentence-transformers 无法从 huggingface.co 下载模型的问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# OpenAI 兼容客户端（用于 DeepSeek）
from openai import OpenAI

# 本地 Embedding
from sentence_transformers import SentenceTransformer

# TextGrad
import textgrad as tg
from textgrad.engine import EngineLM

# 可选绘图
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ============================================================
# 全局配置
# ============================================================

# 从环境变量读取 API Key（提交前务必删除硬编码）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

MODEL_FORWARD = "deepseek-chat"      # 前向生成
MODEL_BACKWARD = "deepseek-chat"     # 反向梯度

# 本地 Embedding 模型（可替换为 BGE-M3 的 API 调用，见注释）
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 若希望使用更强大的本地模型，可改为：
# EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# 记忆检索相似度阈值（降低以提高召回）
SIMILARITY_THRESHOLD = 0.4

# ============================================================
# DeepSeek 客户端
# ============================================================

def get_deepseek_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 全局客户端（用于初始解生成等）
GLOBAL_CLIENT = get_deepseek_client()

# ============================================================
# TextGrad 自定义 Engine
# ============================================================

class DeepSeekEngine(EngineLM):
    """将 DeepSeek 包装为 TextGrad Engine"""
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant that carefully analyzes problems "
        "and provides constructive, detailed feedback."
    )

    def __init__(self, model_string: str = MODEL_BACKWARD,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 temperature: float = 0.0, max_tokens: int = 4096):
        self.model_string = model_string
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = get_deepseek_client()
        self.is_multimodal = False

    def generate(self, content, system_prompt=None, **kwargs):
        sys_prompt = system_prompt or self.system_prompt
        if isinstance(content, str):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": content},
            ]
        else:
            messages = content
        resp = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

def setup_textgrad_with_deepseek():
    engine = DeepSeekEngine()
    tg.set_backward_engine(engine, override=True)
    return engine

# ============================================================
# 数据结构
# ============================================================

@dataclass
class OptimizationExperience:
    problem_id: str
    problem_text: str
    problem_type: str = ""
    initial_solution: str = ""
    final_solution: str = ""
    textual_gradients: List[str] = field(default_factory=list)
    key_insight: str = ""
    num_iterations: int = 0
    success: bool = False
    improvement_score: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

# ============================================================
# 记忆模块（本地 embedding）
# ============================================================

class ExperienceMemory:
    def __init__(self, capacity: int = 1000,
                 embedding_model: str = EMBEDDING_MODEL,
                 similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.experiences: List[OptimizationExperience] = []
        self.embeddings: List[np.ndarray] = []
        self.stats = {"total_stored": 0, "total_retrieved": 0, "cache_hits": 0}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        print(f"[Memory] Loading embedding model: {embedding_model}")
        self._embedder = SentenceTransformer(embedding_model)

    def _get_embedding(self, text: str) -> np.ndarray:
        text = text[:2000]
        cache_key = hash(text)
        if cache_key in self._embedding_cache:
            self.stats["cache_hits"] += 1
            return self._embedding_cache[cache_key]
        emb = self._embedder.encode(text, normalize_embeddings=True)
        self._embedding_cache[cache_key] = emb
        return emb

    def store(self, experience: OptimizationExperience) -> bool:
        if not experience.success:
            return False
        if len(self.experiences) >= self.capacity:
            self.experiences.pop(0)
            self.embeddings.pop(0)
        emb = self._get_embedding(experience.problem_text)
        self.experiences.append(experience)
        self.embeddings.append(emb)
        self.stats["total_stored"] += 1
        return True

    def retrieve(self, query_problem: str, top_k: int = 3,
                 min_similarity: Optional[float] = None) -> Tuple[List[OptimizationExperience], List[float]]:
        if not self.embeddings:
            return [], []
        min_sim = min_similarity or self.similarity_threshold
        query_emb = self._get_embedding(query_problem)
        emb_matrix = np.stack(self.embeddings)
        similarities = emb_matrix @ query_emb
        valid_idx = np.where(similarities >= min_sim)[0]
        if len(valid_idx) == 0:
            return [], []
        sorted_idx = valid_idx[np.argsort(similarities[valid_idx])[::-1]]
        top_idx = sorted_idx[:top_k]
        exps = [self.experiences[i] for i in top_idx]
        sims = [similarities[i] for i in top_idx]
        self.stats["total_retrieved"] += len(exps)
        return exps, sims

    def save(self, filepath: str):
        data = {
            "experiences": [e.to_dict() for e in self.experiences],
            "embeddings": [e.tolist() for e in self.embeddings],
            "stats": self.stats
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.experiences = [OptimizationExperience.from_dict(e) for e in data["experiences"]]
        self.embeddings = [np.array(e) for e in data["embeddings"]]
        self.stats = data["stats"]

# ============================================================
# 记忆增强优化器（修正版：恢复正确的梯度注入逻辑）
# ============================================================

class MemoryAugmentedTGD:
    def __init__(self, parameters: List[tg.Variable], memory: ExperienceMemory,
                 learning_rate: float = 1.0, top_k_experiences: int = 1,
                 augmentation_mode: str = "append", sim_threshold: float = SIMILARITY_THRESHOLD):
        self.parameters = parameters
        self.memory = memory
        self.learning_rate = learning_rate
        self.top_k = top_k_experiences
        self.augmentation_mode = augmentation_mode
        self.sim_threshold = sim_threshold
        self.base_optimizer = tg.TGD(parameters=parameters)
        self.current_problem: Optional[str] = None
        self.gradient_history: List[str] = []
        self.retrieved_experiences: List[OptimizationExperience] = []
        self.retrieved_similarities: List[float] = []

    def set_problem(self, problem_text: str):
        self.current_problem = problem_text
        self.gradient_history = []
        if len(self.memory.experiences) > 0:
            exps, sims = self.memory.retrieve(
                problem_text, top_k=self.top_k,
                min_similarity=self.sim_threshold
            )
            self.retrieved_experiences = exps
            self.retrieved_similarities = sims
        else:
            self.retrieved_experiences = []
            self.retrieved_similarities = []

    def predict_required_iterations(self) -> int:
        if not self.retrieved_experiences:
            return 3
        sim = self.retrieved_similarities[0]
        if sim > 0.6:
            return max(1, min(self.retrieved_experiences[0].num_iterations, 5))
        return 3

    def _format_experience_context(self) -> str:
        if not self.retrieved_experiences:
            return ""
        insight = self.retrieved_experiences[0].key_insight
        if not insight:
            return ""
        return f"\n[Relevant past strategy (sim={self.retrieved_similarities[0]:.2f}): {insight[:200]}]"

    def step(self):
        for param in self.parameters:
            if hasattr(param, "gradients") and param.gradients is not None:
                # 保存原始梯度文本（用于记录历史，不破坏原结构）
                if isinstance(param.gradients, (set, list)):
                    grad_texts = []
                    for g in param.gradients:
                        if hasattr(g, 'value'):
                            grad_texts.append(str(g.value))
                        else:
                            grad_texts.append(str(g))
                    original_gradient = " ".join(grad_texts)
                else:
                    original_gradient = str(getattr(param.gradients, 'value', param.gradients))

                self.gradient_history.append(original_gradient)

                # 记忆增强：向原有的 set 中添加新的梯度 Variable，而不是替换整个 set
                if self.retrieved_experiences:
                    ctx = self._format_experience_context()
                    if ctx:
                        augmented_text = original_gradient + ctx
                        augmented_var = tg.Variable(
                            augmented_text,
                            role_description="augmented textual gradient",
                            requires_grad=False
                        )
                        # 确保 gradients 是 set 类型
                        if not isinstance(param.gradients, set):
                            # 如果原本不是 set，转换为包含原内容的 set
                            param.gradients = set(param.gradients) if isinstance(param.gradients, list) else {param.gradients}
                        # 将增强后的梯度添加到 set 中
                        param.gradients.add(augmented_var)

        # 调用基础优化器（它会正确处理 set 类型的 gradients）
        self.base_optimizer.step()

    def record_success(self, final_solution: str, initial_solution: str):
        if not self.current_problem:
            return
        key_insight = self.gradient_history[-1][:300] if self.gradient_history else "Correct reasoning"
        exp = OptimizationExperience(
            problem_id=str(hash(self.current_problem)),
            problem_text=self.current_problem,
            problem_type=infer_problem_type(self.current_problem),
            initial_solution=initial_solution,
            final_solution=final_solution,
            textual_gradients=self.gradient_history.copy(),
            key_insight=key_insight,
            num_iterations=len(self.gradient_history),
            success=True,
            improvement_score=1.0
        )
        self.memory.store(exp)

# ============================================================
# 辅助函数
# ============================================================

def infer_problem_type(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ['egg', 'duck', 'chicken', 'feed', 'crayon', 'muffin', 'apple']):
        return "arithmetic"
    elif any(k in q for k in ['meter', 'mile', 'hour', 'speed', 'travel', 'mph', 'km']):
        return "distance_rate_time"
    elif any(k in q for k in ['buy', 'sell', 'cost', 'price', 'profit', 'dollar', '$']):
        return "finance"
    elif any(k in q for k in ['box', 'pack', 'group', 'share', 'divide']):
        return "grouping"
    elif any(k in q for k in ['fence', 'area', 'perimeter', 'rectangle', 'garden']):
        return "geometry"
    else:
        return "general_math"

def create_loss_function() -> tg.TextLoss:
    instruction = """
    Evaluate the following mathematical solution.
    Check for correctness, reasoning, and calculation errors.
    If incorrect, explain flaws. If correct, confirm it.
    """
    return tg.TextLoss(instruction)

def generate_initial_solution(question: str) -> str:
    try:
        resp = GLOBAL_CLIENT.chat.completions.create(
            model=MODEL_FORWARD,
            messages=[
                {"role": "system", "content": "You are a math tutor. Solve step by step."},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def extract_numbers(text: str):
    nums = re.findall(r'-?\d+\.?\d*', text)
    return [float(n) for n in nums if n.strip()]

def check_answer(solution: str, ground_truth: str) -> bool:
    nums_sol = extract_numbers(solution)
    nums_gt = extract_numbers(ground_truth)
    if not nums_sol or not nums_gt:
        return False
    return abs(nums_sol[-1] - nums_gt[-1]) < 0.01

# ============================================================
# 单题运行
# ============================================================

def run_single_problem(problem: dict, method: str,
                       memory: Optional[ExperienceMemory],
                       max_iterations: int = 5,
                       is_training: bool = False) -> dict:
    question = problem["question"]
    ground_truth = problem["answer"]

    start_total = time.time()
    initial_solution = generate_initial_solution(question)
    init_time = time.time() - start_total

    solution = tg.Variable(initial_solution,
                           role_description="step-by-step math solution",
                           requires_grad=True)
    loss_fn = create_loss_function()

    if method == "vanilla":
        optimizer = tg.TGD(parameters=[solution])
        actual_max_iters = max_iterations
        predicted_iters = None
    else:
        optimizer = MemoryAugmentedTGD(
            parameters=[solution],
            memory=memory,
            top_k_experiences=1,
            sim_threshold=SIMILARITY_THRESHOLD
        )
        optimizer.set_problem(question)
        predicted_iters = optimizer.predict_required_iterations()
        actual_max_iters = min(max_iterations, predicted_iters + 1)

    api_calls = 1  # 初始解
    loop_start = time.time()
    success = False
    final_solution = initial_solution
    num_iters = 0
    iterations_log = []

    for i in range(actual_max_iters):
        if check_answer(solution.value, ground_truth):
            success = True
            final_solution = solution.value
            num_iters = i
            break

        loss = loss_fn(solution)
        api_calls += 1
        loss.backward()
        api_calls += 1

        iterations_log.append({
            "iteration": i+1,
            "solution": solution.value[:200],
            "gradient": str(solution.gradients)[:200] if solution.gradients else ""
        })

        optimizer.step()
        api_calls += 1

    if not success:
        final_solution = solution.value
        num_iters = actual_max_iters
        success = check_answer(solution.value, ground_truth)

    loop_time = time.time() - loop_start
    total_time = init_time + loop_time

    if is_training and success and method == "mat":
        optimizer.record_success(final_solution=final_solution,
                                 initial_solution=initial_solution)

    return {
        "method": method,
        "question": question[:200],
        "initial_solution": initial_solution,
        "final_solution": final_solution,
        "success": success,
        "num_iterations": num_iters,
        "api_calls": api_calls,
        "time": total_time,
        "predicted_iter": predicted_iters if method == "mat" else None,
        "iterations_log": iterations_log
    }

# ============================================================
# 两阶段实验
# ============================================================

def run_two_phase_experiment(train_problems: List[dict],
                             test_problems: List[dict],
                             max_iterations: int = 3):
    print("=" * 70)
    print("🚀 两阶段实验：训练(积累记忆) → 测试(对比泛化)")
    print("=" * 70)

    # ---------- 阶段1：训练 MAT ----------
    print("\n📚 阶段1：训练 MAT (积累记忆)")
    print("-" * 50)
    memory = ExperienceMemory(capacity=500)
    for i, prob in enumerate(train_problems):
        print(f"训练 {i+1}/{len(train_problems)}: {prob['question'][:50]}...")
        res = run_single_problem(prob, method="mat", memory=memory,
                                 max_iterations=max_iterations, is_training=True)
        status = "✓" if res["success"] else "✗"
        print(f"  {status} 迭代: {res['num_iterations']}, API: {res['api_calls']}, 耗时: {res['time']:.2f}s")

    print(f"\n📊 训练完成，记忆库存储 {memory.stats['total_stored']} 条成功经验")

    # ---------- 阶段2：测试对比 ----------
    print("\n🧪 阶段2：测试对比 (Vanilla vs MAT)")
    print("-" * 50)
    all_test_results = {"vanilla": [], "mat": []}

    for method in ["vanilla", "mat"]:
        print(f"\n--- 测试方法: {method.upper()} ---")
        for i, prob in enumerate(test_problems):
            print(f"测试 {i+1}/{len(test_problems)}: {prob['question'][:50]}...")
            res = run_single_problem(prob, method=method,
                                     memory=memory if method == "mat" else None,
                                     max_iterations=max_iterations, is_training=False)
            all_test_results[method].append(res)
            status = "✓" if res["success"] else "✗"
            pred = res.get("predicted_iter")
            pred_str = f"预测={pred}" if pred is not None else ""
            print(f"  {status} 迭代: {res['num_iterations']} {pred_str} API: {res['api_calls']} 耗时: {res['time']:.2f}s")

    # ---------- 汇总报告 ----------
    print("\n" + "=" * 70)
    print("📈 测试阶段最终对比")
    print("=" * 70)

    summary = {}
    for method, res_list in all_test_results.items():
        successes = sum(1 for r in res_list if r["success"])
        total = len(res_list)
        iters = [r["num_iterations"] for r in res_list]
        apis = [r["api_calls"] for r in res_list]
        times = [r["time"] for r in res_list]
        summary[method] = {
            "accuracy": successes / total,
            "avg_iterations": np.mean(iters),
            "avg_api_calls": np.mean(apis),
            "avg_time": np.mean(times)
        }
        print(f"\n{method.upper()}:")
        print(f"  准确率: {successes}/{total} ({100*successes/total:.1f}%)")
        print(f"  平均迭代次数: {np.mean(iters):.2f}")
        print(f"  平均 API 调用: {np.mean(apis):.2f}")
        print(f"  平均耗时 (秒): {np.mean(times):.2f}")

    # 保存详细结果
    with open("phase2_test_results.json", "w", encoding="utf-8") as f:
        json.dump(all_test_results, f, indent=2, ensure_ascii=False)
    memory.save("memory_after_training.json")
    print("\n✅ 详细结果已保存至 phase2_test_results.json 和 memory_after_training.json")

    # 绘制对比图
    if HAS_PLT:
        plot_comparison(summary)
    else:
        print("⚠️ matplotlib 未安装，跳过绘图。")

    return all_test_results, summary

def plot_comparison(summary: dict):
    """绘制 Vanilla vs MAT 的柱状对比图"""
    methods = list(summary.keys())
    acc = [summary[m]["accuracy"] * 100 for m in methods]
    iters = [summary[m]["avg_iterations"] for m in methods]
    apis = [summary[m]["avg_api_calls"] for m in methods]
    times = [summary[m]["avg_time"] for m in methods]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    axes[0].bar(methods, acc, color=['skyblue', 'salmon'])
    axes[0].set_title("Accuracy (%)")
    axes[0].set_ylim(0, 100)

    axes[1].bar(methods, iters, color=['skyblue', 'salmon'])
    axes[1].set_title("Avg Iterations")

    axes[2].bar(methods, apis, color=['skyblue', 'salmon'])
    axes[2].set_title("Avg API Calls")

    axes[3].bar(methods, times, color=['skyblue', 'salmon'])
    axes[3].set_title("Avg Time (s)")

    plt.tight_layout()
    plt.savefig("comparison_plot.png", dpi=150)
    print("📊 对比图已保存为 comparison_plot.png")

# ============================================================
# 题目数据集
# ============================================================

TRAIN_PROBLEMS = [
    {"question": "Janet's ducks lay 16 eggs per day. She eats 3 and uses 4 for muffins. Sells the rest at $2 each. Daily earnings?", "answer": "18"},
    {"question": "A robe takes 2 bolts blue fiber and half that white. Total bolts?", "answer": "3"},
    {"question": "Josh buys house for $80k, repairs $50k, value up 150%. Profit?", "answer": "70000"},
    {"question": "James runs 3 sprints 3x/week, 60m each. Total meters per week?", "answer": "540"},
    {"question": "Wendi has 20 chickens, 3 cups/day each. Morning 15, afternoon 25. Final meal cups?", "answer": "20"},
    {"question": "Tom has 8 boxes of 24 crayons. Gives 15 to sister, rest to 5 friends. Each friend gets?", "answer": "37"},
    {"question": "Bakery sells muffins in packs of 4 and 6. Sarah buys 3 packs of 4 and 2 of 6. Total?", "answer": "24"},
    {"question": "Theater: 12 rows of 18 seats. 157 occupied. Empty seats?", "answer": "59"},
    {"question": "Lisa reads 240 pages: 25 Mon, 30 Tue, twice Tue on Wed. Pages left?", "answer": "125"},
    {"question": "Farmer has 150 apples, sells 45, divides rest into 7 baskets. Apples per basket?", "answer": "15"},
]

TEST_PROBLEMS = [
    {"question": "Michael earns $12 per hour. Works 8h Mon, 6h Tue, 9h Wed. Total earnings?", "answer": "276"},
    {"question": "Rectangular garden 15m by 8m. Fencing costs $9 per meter. Total cost?", "answer": "414"},
    {"question": "Emma has 120 stickers. Gives 1/3 to brother, buys 25 more. How many now?", "answer": "105"},
    {"question": "Car travels at 65 mph for 3.5 hours. Distance?", "answer": "227.5"},
    {"question": "28 students split into groups of 4. How many groups?", "answer": "7"},
    {"question": "Store sells pencils in packs of 10 and 15. John buys 4 packs of 10 and 2 of 15. Total?", "answer": "70"},
    {"question": "Pizza cut into 8 slices. 3 friends eat 2 slices each. Left?", "answer": "2"},
    {"question": "Train travels 300 miles in 5 hours. Average speed?", "answer": "60"},
    {"question": "Samantha has $50. Buys book $18 and toy $22. Left?", "answer": "10"},
    {"question": "School orders 15 boxes of pencils, 24 per box. Gives 100 to students. Left?", "answer": "260"},
    {"question": "Baker makes 240 cookies, packs in bags of 8. How many bags?", "answer": "30"},
    {"question": "Movie starts 7:30pm, lasts 2h 15m. End time?", "answer": "9:45"},
    {"question": "Rectangle length 12cm, width 5cm. Area?", "answer": "60"},
    {"question": "Tank holds 500L, filled at 25L/min. Time to fill?", "answer": "20"},
    {"question": "Lily reads 35 pages/day. Pages in 12 days?", "answer": "420"},
    {"question": "Shirt costs $25, 20% discount. Sale price?", "answer": "20"},
    {"question": "Bus has 45 seats, 38 occupied. Empty seats?", "answer": "7"},
    {"question": "David runs 5km in 25 min. Speed in km/h?", "answer": "12"},
    {"question": "Cake recipe needs 3 eggs. Eggs for 5 cakes?", "answer": "15"},
    {"question": "Phone costs $600, 15% off. Discount amount?", "answer": "90"},
    {"question": "John saves $50 per week. After 8 weeks, buys $120 bike. Left?", "answer": "280"},
    {"question": "Class has 18 boys and 12 girls. Ratio boys to total?", "answer": "0.6"},
    {"question": "Book has 350 pages. Read 120 pages. Pages left to read 50%?", "answer": "175"},
    {"question": "Fruit basket: 8 apples, 6 oranges, 4 bananas. Fraction of apples?", "answer": "0.44"},
    {"question": "Train leaves at 9:15, arrives 11:45. Travel time in minutes?", "answer": "150"},
    {"question": "Painter paints 3 rooms in 2 days. Rooms in 10 days?", "answer": "15"},
    {"question": "Water bill $45, electric bill 2.5 times water. Total?", "answer": "157.5"},
    {"question": "Garden area 120 sq ft, length 12 ft. Width?", "answer": "10"},
    {"question": "Sale: 30% off $80 jacket. Final price?", "answer": "56"},
    {"question": "Recipe uses 2 cups flour for 12 muffins. Flour for 30 muffins?", "answer": "5"},
]

# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    # 初始化 TextGrad 引擎
    print("[Setup] Configuring DeepSeek engine for TextGrad...")
    setup_textgrad_with_deepseek()

    # 运行两阶段实验
    run_two_phase_experiment(TRAIN_PROBLEMS, TEST_PROBLEMS, max_iterations=3)

