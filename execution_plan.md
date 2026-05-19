# 工业级 RL 环境缺陷修复与自动化评测基准构建 — 执行计划

## 1. 总体路线图

项目分为 6 个阶段推进，总计预计约 2-3 周（视实际编码进度而定）。

```
Phase 1: 项目骨架搭建       → Day 1
Phase 2: Task 1 库存管理     → Day 2-3
Phase 3: Task 2 作业调度     → Day 4-6
Phase 4: Task 3 广告竞价     → Day 7-10
Phase 5: Task 4 缓存策略     → Day 11-14
Phase 6: 集成测试与文档完善  → Day 15-16
```

---

## 2. Phase 1：项目骨架搭建

### 目标
建立项目目录结构、Docker 环境和公共配置，确保后续任务可以在统一的环境中开发测试。

### 具体任务
- [ ] 创建完整目录树（tasks/01~04、verify/）
- [ ] 编写 Dockerfile（基于 python:3.10-slim）
- [ ] 编写 docker-compose.yml
- [ ] 编写 requirements.txt（锁定 gymnasium, numpy, torch, stable-baselines3 等版本）
- [ ] 编写 verify/verify_all.sh（串行执行所有任务的 solve.sh）
- [ ] 编写 .gitignore

### 关键输出
- Dockerfile + docker-compose.yml：`docker compose up` 即可进入开发环境
- requirements.txt：所有依赖版本锁定的 pip 依赖文件

### 验证方式
```bash
docker compose build
docker compose run --rm app python -c "import gymnasium; print(gymnasium.__version__)"
```

---

## 3. Phase 2：Task 1 — Inventory Management（库存管理）

### 目标
实现最简单的梯度任务，快速验证完整 pipeline（问题代码 → 修复 → 测试 → 验证）。

### 具体任务
- [ ] 编写 `tasks/01_inventory_management/environment/env.py`（含 3 个缺陷的 Gymnasium 环境）
- [ ] 编写 `tasks/01_inventory_management/solution/fixed_env.py`（修复后的环境）
- [ ] 编写 `tasks/01_inventory_management/solution/train.py`（基于 PPO 的最短训练流程）
- [ ] 编写 `tasks/01_inventory_management/solution/tests/test_env.py`（黑盒测试）
- [ ] 编写 `tasks/01_inventory_management/solution/solve.sh`（一键验证脚本）
- [ ] 编写 `tasks/01_inventory_management/README.md`（纯英文任务描述）

### 缺陷清单
| 缺陷 | 定位 | 修复方式 |
|------|------|----------|
| 奖励符号反转 | `reward = holding_cost + stockout_penalty` | 改为 `reward = -(holding_cost + stockout_penalty)` |
| 库存计算错误 | `ending_inv = starting_inv + order` | 改为 `ending_inv = starting_inv + order - demand` |
| 永不终止 | 未实现 `truncated` 条件 | 添加 `max_steps` 限制和步骤计数 |

### 验证方式
```bash
cd tasks/01_inventory_management && bash solution/solve.sh
```

---

## 4. Phase 3：Task 2 — Job Shop Scheduling（作业调度）

### 目标
实现中等难度的调度任务，引入动作屏蔽和数值稳定性方面的缺陷。

### 具体任务
- [ ] 编写 `tasks/02_job_scheduling/environment/env.py`（含 3 个缺陷）
- [ ] 编写 `tasks/02_job_scheduling/solution/fixed_env.py`
- [ ] 编写 `tasks/02_job_scheduling/solution/train.py`
- [ ] 编写 `tasks/02_job_scheduling/solution/tests/test_env.py`
- [ ] 编写 `tasks/02_job_scheduling/solution/solve.sh`
- [ ] 编写 `tasks/02_job_scheduling/README.md`

### 缺陷清单
| 缺陷 | 定位 | 修复方式 |
|------|------|----------|
| 动作屏蔽缺失 | `step()` 中接受已完成的作业 | 添加 `action_masks`，过滤已完成作业 |
| 提前终止 | `step_count` 初始值/检查时机错误 | 调整 step_count 初始化为 1 或修正检查点为 `>` |
| 奖励归一化爆炸 | `reward = raw_reward / small_number` | 添加数值保护 `max(small_number, epsilon)` |

---

## 5. Phase 4：Task 3 — Ad Bidding / Campaign Optimization（广告竞价优化）

### 目标
实现中等偏高难度的预算分配任务，引入预算约束、状态设计和非平稳性等方面的缺陷。

### 具体任务
- [x] 编写 `tasks/03_ad_bidding/environment/env.py`（含 4 个缺陷）
- [x] 编写 `tasks/03_ad_bidding/solution/fixed_env.py`
- [x] 编写 `tasks/03_ad_bidding/solution/train.py`
- [x] 编写 `tasks/03_ad_bidding/solution/tests/test_env.py`
- [x] 编写 `tasks/03_ad_bidding/solution/solve.sh`
- [x] 编写 `tasks/03_ad_bidding/README.md`

### 缺陷清单
| 缺陷 | 定位 | 修复方式 |
|------|------|----------|
| 预算泄漏 | `step()` 中未校验总花费 ≤ 总预算 | 添加 spend 检查并 clip 动作 |
| 状态缺失 | `observation` 缺少 `remaining_budget` | 将 remaining_budget 添加到观测向量 |
| 非平稳奖励未归一化 | 奖励直接使用原始转化数 | 添加 running mean/std 进行奖励归一化 |
| 种子未固定 | `np.random.seed()` 未调用 | 在 `reset()` 中固定所有随机种子 |

---

## 6. Phase 5：Task 4 — Cache Replacement Policy（缓存替换策略）

### 目标
实现最高难度的 CDN 缓存策略任务，缺陷最隐蔽、涉及面最广。

### 具体任务
- [ ] 编写 `tasks/04_cache_policy/environment/env.py`（含 4 个缺陷）
- [ ] 编写 `tasks/04_cache_policy/solution/fixed_env.py`
- [ ] 编写 `tasks/04_cache_policy/solution/train.py`
- [ ] 编写 `tasks/04_cache_policy/solution/tests/test_env.py`
- [ ] 编写 `tasks/04_cache_policy/solution/solve.sh`
- [ ] 编写 `tasks/04_cache_policy/README.md`

### 缺陷清单
| 缺陷 | 定位 | 修复方式 |
|------|------|----------|
| 状态缺失时间特征 | `_get_obs()` 只返回内容 ID | 添加 access_count, time_since_last_access, frequency 等特征 |
| 延迟奖励累加错误 | 多步奖励累加逻辑混乱 | 修正为每步立即计算 hit/miss 奖励 |
| 内存泄漏 | `self.history.append(...)` 持续增长 | 限制 history 最大长度或用 deque |
| 终止条件变量错误 | 判断使用了 `steps` 而非 `self.current_step` | 修正变量引用 |

---

## 7. Phase 6：集成测试与文档完善

### 目标
全量验证所有任务，完善英文文档和项目级文档。

### 具体任务
- [ ] 运行 `verify/verify_all.sh` 全量验证
- [ ] 确认每个任务的 buggy 环境在测试中确实失败
- [ ] 确认每个任务的 fixed 环境在测试中全部通过
- [ ] 完善 `TASK.md` 纯英文总任务描述
- [ ] 完善 `execution_plan.md` 中文执行计划
- [ ] 最终 README.md 项目说明

### 全量验证
```bash
# 构建 Docker
docker compose build

# 运行全量验证
docker compose run --rm app bash verify/verify_all.sh
```

预期输出：所有任务通过，退出码 0。

---

## 8. 技术要点

### 8.1 Gymnasium 环境设计规范

所有环境遵循 Gymnasium Env 接口：

```python
import gymnasium as gym
from gymnasium import spaces

class MyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(...)
        self.action_space = spaces.Discrete(...)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # ... 初始化逻辑
        return obs, info

    def step(self, action):
        # ... 环境逻辑
        return obs, reward, terminated, truncated, info
```

### 8.2 黑盒测试要点

测试仅通过 Gymnasium API 进行交互，不访问内部属性：
- 检查 `observation_space` 和 `action_space` 的 shape/dtype
- 检查 `reset()` 返回的 obs 在 `observation_space` 内
- 检查 `step()` 返回的 obs 在 `observation_space` 内
- 检查奖励范围是否合理
- 运行 100 个 episode 确保无崩溃
- 验证相同 seed 返回相同结果

### 8.3 缺陷设计原则

- 所有缺陷在代码层面是合理的（非恶意代码）
- 有缺陷的环境可正常运行，不会抛出未处理异常
- 训练指标明显低于修复后的版本（可区分性）
- 每个缺陷都可以通过黑盒测试检测出来

---

## 9. 依赖与版本

| 包名 | 版本约束 |
|------|----------|
| python | >= 3.10 |
| gymnasium | >= 0.29, < 1.1 |
| numpy | >= 1.24, < 2.0 |
| torch | >= 2.0, < 3.0 |
| stable-baselines3 | >= 2.0 |
| pytest | >= 7.0 |

---

## 10. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Buggy 环境意外无法运行 | 测试流程断裂 | 每个缺陷编写后单独测试可运行性 |
| Docker 镜像过大 | 超过 2GB 约束 | 使用 slim 镜像，最小化依赖 |
| 缺陷区分度不够 | 模型都能修复，无法区分能力 | 每个 Task 包含一个"陷阱"型缺陷 |
| Gymnasium 版本兼容性 | API 变更导致环境报错 | 锁定版本并在 CI 中验证 |
