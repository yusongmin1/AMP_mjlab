# mjlab 项目问答笔记

> 记录对 `AMP_mjlab-main` 与 `DroidUpE1_mjlab-main` 两个仓库的分析问答，持续追加。

---

## Q1: AMP_mjlab 为什么要打 mjlab 补丁，DroidUpE1 却不用？

**结论：DroidUpE1 不需要补丁，因为它没用到 `time` 排序功能；AMP_mjlab 用到了，必须打补丁。**

### 补丁改了什么

补丁文件 `AMP_mjlab-main/mjlab_patch/mjlab/managers/observation_manager.py` 给 `ObservationGroupCfg` 新增了 `history_ordering` 参数（原版 mjlab 只有 `term` 一种排序）：

- `"term"`（默认，原版行为）：每个 term 的历史先各自展平再拼接 → `[A_t0, A_t1, ..., B_t0, B_t1, ...]`
- `"time"`（补丁新增）：同一时刻的所有 term 挨在一起 → `[A_t0, B_t0, ..., A_t1, B_t1, ...]`

### AMP_mjlab 为什么必须打

`src/tasks/amp_loco/amp_env_cfg.py` 中 actor/critic group 显式配置了 `history_ordering="time"`，不打补丁会直接报未知参数错误。且 actor group 有多个 term（`base_ang_vel`、`joint_pos`、`joint_vel`、`actions`、`height_scan` 等），多 term + history 时 time-major 和 term-major 排出的向量顺序完全不同。

### DroidUpE1 为什么不需要

- 它 `.venv` 里安装的原版 `observation_manager.py` 没有 `history_ordering` 参数；
- 它自己的配置也没用这个参数，只用 `history_length=5`；
- 关键设计差异：它把所有观测在函数里先拼成**一整个 72 维大向量**（`actor_frame`），作为**单个 term** 传给 manager。只有一个 term 时，time-major 和 term-major 的展开结果完全一样，原版 mjlab 默认排序就够用。

### 对比总结

| | AMP_mjlab（多 term + patch） | DroidUpE1（单 frame term） |
|---|---|---|
| obs 组织 | 每类观测一个 term，manager 负责拼接 | 函数内 `torch.cat` 成一帧 72 维，manager 只见一个 term |
| history 布局 | 需要 patch 加 `history_ordering="time"` | 单 term，默认展平就是逐帧，无需 patch |
| 噪声/尺度 | 享受 manager 的 per-term `noise`/`scale`/`clip`/`delay` 配置 | 在 obs 函数里手工写字段级噪声 |
| 部署对齐 | 依赖 patch 后的 time-major 布局约定 | `(H, 72)` 数组滚动 + reshape，天然一致，ONNX metadata 校验 |

两者殊途同归：最终给网络的观测都是"按时间排的整帧历史"。区别只是实现层次——AMP_mjlab 让 manager 支持任意多 term 的 time-major 交错（灵活但要改源码），DroidUpE1 把布局责任前移到观测函数里，用"单 term"约束换零改动。代价是失去 manager 的 per-term 便捷配置（噪声得手写），以后想给某类观测单独加 delay/history 不方便。

---

## Q2: DroidUpE1 的"单 frame term"是怎么设计的？

**核心：整帧打包成单一 term，把补丁要解决的问题在更高一层绕开。**

### 1. 观测函数自己拼帧

`src/tasks/amp/mdp/observations.py` 的 `_actor_frame` 内部把角速度、重力向量、速度指令、关节位置/速度、上一步动作全部 `torch.cat` 成一个 72 维向量（`3+3+3+21+21+21 = 72`，常量定义在 `src/tasks/amp/constants.py` 的 `ACTOR_FRAME_DIM`）。整个 actor group 只有一个 term，历史就是这个 72 维向量的 5 次采样。

### 2. 噪声也是为这个设计服务的

单 term 没法用 manager 的 per-term noise，所以噪声直接写在函数里、按 frame 内字段区间手工加：

```python
scale = torch.zeros(ACTOR_FRAME_DIM, device=env.device)
scale[0:3] = 0.3 * 0.2      # ang_vel
scale[3:6] = 0.05           # gravity
scale[9:30] = 0.02          # joint_pos
scale[30:51] = 1.5 * 0.05   # joint_vel
frame = frame + (2.0 * torch.rand_like(frame) - 1.0) * scale
```

源码注释明确写着："Keeping the whole frame in one observation term preserves frame-major history order"——刻意设计，为了保住"帧优先"的历史布局。

### 3. 部署侧和训练侧天然对齐

`sim2sim/sim2sim_e1_21dof_amp.py` 里历史缓冲是 `(5, 72)` 数组，每步滚动一行再 `reshape(-1)`，C-order 的 reshape 就是 time-major，和训练时单 term 展平结果逐位一致。另外导出 ONNX 时把 `actor_history_length=5`、`actor_frame_dim=72` 写进 metadata，sim2sim 加载时强制校验，不匹配直接报错。

---

## Q3: DroidUpE1（单 frame term）有多少帧数据？

"帧"有两种含义：

### 1. 观测历史帧（policy 输入）：5 帧

`history_length=5`，每帧 72 维，拼成 360 维观测（72×5=360，写在 sim2sim 文件头注释里）。

### 2. AMP 动作数据集帧数

训练配置 `src/tasks/amp/config/e1_21dof/rl_cfg.py` 的 `amp_motion_files` 实际加载 8 个 NPZ：

| 文件 | 帧数 | fps | 时长 | 权重 |
|---|---|---|---|---|
| `walk_moving.npz` | 11,535 | 100 | 115.35s | 1.0 |
| `run.npz` | 1,064 | 50 | 21.28s | 0.5 |
| `run_mirror.npz` | 1,064 | 50 | 21.28s | 0.5 |
| `turn_l.npz` | 1,170 | 60 | 19.50s | 0.5 |
| `turn_r.npz` | 1,041 | 60 | 17.35s | 0.5 |
| `side_l.npz` | 2,045 | 100 | 20.45s | 0.5 |
| `side_r.npz` | 1,757 | 100 | 17.57s | 0.5 |
| `stand.npz` | 130 | 100 | 1.30s | 1.0 |
| **合计** | **19,806 帧** | — | **≈234s** | — |

8 段对应 5 种风格标签（`walk, run, turn, side, stand`，判别器拼 one-hot 条件向量用）。各文件 fps 不同（50/60/100），motion_loader 按 fps 归一化到控制频率采样，总时长 ≈234 秒才是有效量。

另外 `dataset/` 还有未被训练配置引用的文件（`walk.npz` 11,685 帧、`side.npz` 4,249 帧、`turn.npz` 3,072 帧等，原始/备选片段），以及 mimic 任务数据（backflip 150 帧、三个舞蹈共 3,611 帧）。

---

## Q4: AMP_mjlab 的恢复窗口是怎么创建的？

**恢复窗口 = 三个组件配合：延迟终止 + 计数器 + 恢复数据重置。**

### 整体流程

```
robot 摔倒 → termination 触发 done
           → DelayedTerminationManager 拦下 reset 信号（窗口开始）
           → 机器人躺在地上继续跑 250 个 policy step（≈5 秒）自生自灭
                ├─ 期间自己爬起来了 → done 变 False → 计数器清零，episode 继续活
                └─ 250 步后还没起来 → 放行 reset
                                  → reset 从 Recovery 动作随机帧重置（往往是倒地姿态）
```

### 1. 启动时：划分"延迟 envs"并替换 termination manager

`src/tasks/amp_loco/config/g1/env_cfgs.py`：训练时 40% envs 延迟重置（play 时 100%），窗口 250 步：

```python
cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.4
cfg.events["init_motion_loader"].params["max_delay_steps"] = 250
cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir      # WalkandRun
cfg.events["init_motion_loader"].params["recovery_dir"] = _recovery_dir  # Recovery
```

`init_motion_loader`（startup 事件，在 `mdp/events.py`）随机抽 40% env 打上 `_delay_env_mask` 标记，然后用 `DelayedTerminationManager` 包装替换 `env.termination_manager`（`self.__dict__.update(base.__dict__)` 偷取内部状态）。

### 2. 窗口核心：DelayedTerminationManager 计数器（`mdp/terminations.py`）

```python
delay_and_done = self._delay_env_mask & dones
self._delay_counters[delay_and_done] += 1

not_ready = delay_and_done & (self._delay_counters < self._max_delay_steps)
self._terminated_buf[not_ready] = False   # 拦下 reset

ready = delay_and_done & (self._delay_counters >= self._max_delay_steps)
self._delay_counters[ready] = 0           # 超时放行

self._delay_counters[self._delay_env_mask & ~dones] = 0  # 自己爬起来→清零免死
```

关键点：

- **窗口时长**：policy 频率 = timestep 0.005 × decimation 4 = 50Hz，250 步 = **5 秒**。
- **按"连续 done 步数"累加**：一直躺着每步 +1，第 250 步才放行 `terminated`。
- **自救成功免死**：done 变 False 则计数器清零，episode 存活——这是"学会自救"的回报空间来源。
- 非 delay 的 60% env 不受影响，摔倒立刻重置（正常 AMP 训练节奏）。

### 3. 窗口结束后：从 Recovery 数据随机采帧重置

reset 事件 `reset_from_motion_data` 查询单例 `MotionResetManager`，按 delay mask 把 env 分两拨：

- 正常 envs → 从 `WalkandRun` 帧库随机采帧重置；
- delay envs → 从 `Recovery` 帧库随机采帧重置（无 Recovery 数据时回退 WalkandRun）。

`_write_reset_state` 写 root pose / root velocity / 关节状态（root z 叠加地形高度，关节角 clamp 到软限位），delay env 的新 episode 有相当概率从"躺地上"姿态开始，和上一条命结束时的状态分布接得上，形成连续的"摔倒→爬起"经验。

### 数据规模（实测）

| 数据集 | 文件数 | 总帧数 | 时长 |
|---|---|---|---|
| `WalkandRun/` | 17 段 | 7,417 帧 | 148.3s |
| `Recovery/`（fallAndGetUp1_subject1.npz） | 1 段 | 2,575 帧 | 51.5s |

### 细节：AMP 判别器与恢复数据共用

`ampmotion_loader.py` 的 `get_full_frame_batch` 用 `motion_data + motion_data_recovery` 两套拼接采样，即 fall-and-get-up 动作风格同时通过 AMP 判别器损失约束策略——reset 采样和判别器先验用同一份恢复数据，两条路配套。

---

## Q5: 0.4 的恢复 env 是固定的机器人还是每回合随机采样？

**结论：env 身份固定（训练启动时抽一次，终身不变），每回合随机的只是采的恢复帧。**

### 1. mask 只在启动时采样一次

`init_motion_loader` 是 `mode="startup"` 事件，整个训练只跑一次。`torch.randperm` 抽一次 40% 的 env 索引写进 mask，交给 `DelayedTerminationManager` 保存：

```python
num_delay = int(env.num_envs * delay_reset_env_ratio)
delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
delay_indices = torch.randperm(env.num_envs, device=env.device)[:num_delay]
delay_mask[delay_indices] = True
env.termination_manager = DelayedTerminationManager(
    base=env.termination_manager,
    delay_env_mask=delay_mask,
    max_delay_steps=max_delay_steps,
)
```

### 2. 之后 mask 只被读取，从不重采样

全仓库搜 `_delay_env_mask`，它只被 4 处**读取**、无任何重新赋值：

- `terminations.py` — 判断要不要拦 done 信号
- `events.py` 的 reset — 判断从哪个帧库采重置帧
- `rewards.py` — 给 delay envs 单独算恢复相关 reward shaping
- `metrics.py` — 统计恢复成功率指标

所以训练开始后，比如 4096 env 里的 1638 个（随机选出的索引）**永远**是"延迟 env"，其余**永远**是"立即重置 env"，直到训练结束。

### 3. 每回合随机的是"帧"，不是"身份"

每个 delay env 每次 reset 时从 Recovery 帧库（2,575 帧）里 `torch.randint` 随机采一帧——这一帧每回合不同，但 env 属于 delay 组的身份不变。

| 量 | 固定 or 随机 | 频率 |
|---|---|---|
| 哪些 env 是 delay env（40%） | 固定 | 训练启动时定一次，终身不变 |
| delay env 重置时采的恢复帧 | 随机 | 每次 reset 重新采 |

### 细节：reset 帧来源只看 mask 身份，不看死因

delay env 即使因为 20 秒超时（truncation）正常结束，也会从 Recovery 倒地数据采帧开始新回合。这是"让 delay envs 高频接触倒地状态"的刻意设计，代价是这些 env 的平均初始状态质量比正常 env 差一些。

---

## Q6: 要是一直摔倒爬不起来，是不是就在一直练习倒地恢复？

**结论：是的，而且这是刻意设计的训练机制——失败循环里的奖励结构专门为"练习爬起"做了配平，并非浪费算力。**

### 一个爬不起来的 delay env 的完整循环

```
摔倒 → done 被拦截 → 5 秒窗口（躺在地上）
     │   期间：速度跟踪奖励 = 0（ratio 0.0）
     │         站起来奖励 = 3.5 倍加权（唯一的正任务奖励）
     │         is_terminated 惩罚被抑制（terminated_buf 被改成 False）
     ├─ 爬起来了 → 计数器清零，episode 继续，免 -200 惩罚
     └─ 5 秒没爬起来 → 放行 reset → 吃一次 -200
                    → 从 Recovery 随机帧（大概率又是倒地/半倒姿态）开始新回合
                    → 换个姿势再练一次爬起 → 循环
```

### 1. 窗口期间任务奖励被"切换"成恢复模式

`rewards.py` 的 `_get_delay_env_mask` 返回 `delay_env_mask & (delay_counters > 0)`——只在机器人真正倒下后（计数器 > 0）才激活，不是所有 delay env 永久生效。配合奖励配置：

- `track_anchor_linear/angular_velocity`、`body_ang_vel_xy_l2`：`delay_env_rew_ratio=0.0` → 躺着时速度奖励清零（躺着追速度指令无意义）；
- `track_root_height`：`mask_only=True` + `delay_env_rew_ratio=3.5` → 普通 env 给 0，**窗口期唯一正任务奖励**，内容是把 root 拉回默认站立高度——"爬起来"的显式奖励信号，3.5 倍加权。

### 2. -200 终止惩罚在窗口内被"冻结"

`is_terminated` 奖励读 `termination_manager.terminated`（即 `_terminated_buf`），而 `DelayedTerminationManager.compute()` 恰好把窗口期的 `_terminated_buf` 改成 False。所以：

- 窗口内每步惩罚为 0（否则 -200×250 步 = -50,000 会毁掉训练）；
- 只有 5 秒超时放行那一步吃一次 -200；
- 自救成功 = 完全免罚 + episode 继续积累后续奖励 → 强激励"在窗口内爬起来"。

### 3. AMP 判别器也在教它怎么爬

判别器参考数据是 `WalkandRun + Recovery` 拼接采样，窗口期内策略输出的爬起动作也能拿到 AMP 风格奖励——不只被 root height 奖励引导，还被示范动作"手把手"教爬起的姿势序列。

### 4. "一直失败"的 env 不是浪费

- 这 40% env 就是专职生产恢复经验的：失败循环每一步都在"倒地状态 → 尝试爬起"的转移上采样；
- 另外 60% env 同步在练走路，共享同一个策略网络——课程并行不互斥；
- 策略变好后循环自然瓦解：自救成功 → 计数器清零 → episode 存活 → 回到正常训练。README 说的"2 万 iter 左右突然学会摔倒爬起、指标跳变"就是这个循环大规模瓦解的时刻；
- `metrics.py` 的 `mean_delay_steps`（平均躺的步数）就是监控恢复快慢的指标。

### 代价

爬不起来的 env 每 ~5 秒才 reset 一次，经验分布里"倒地姿态"占比极高、"正常行走"占比低——所以只给 40% 而不是全部 env 开延迟；play 模式才开 100%（纯展示恢复能力）。

---

## Q7: terminations.py 第 49-56 行（ready 清零 / 非_done 清零 / return）的意义？

**结论：这是恢复窗口的"收尾三件事"——超时放行并重置计时器、自救赦免、返回篡改后的真实 done 信号。**

### 第 49-51 行：超时 → 放行 reset + 计时器归零

```python
ready = delay_and_done & (self._delay_counters >= self._max_delay_steps)
self._delay_counters[ready] = 0
```

- `ready` 的 env 没被上一段 `_terminated_buf[not_ready] = False` 覆盖——terminated 保持 True，这一步真的 reset（"放行"通过"不拦截"实现）。
- `counter=0` 为什么必须有：reset 从 **Recovery 帧库**采帧，大概率又是倒地姿态 → 下一步又 done → counter=1 → 重新拦截 → 新一轮完整 5 秒窗口。
- 不清零的 bug：counter 仍是 250 → 下一步 done 变 251 ≥ 250 → 立即二次 reset → "重置到倒地姿态却没机会爬起"的死循环。

### 第 53-54 行：自救赦免——定义 counter 语义为"连续 done 步数"

```python
self._delay_counters[self._delay_env_mask & ~dones] = 0
```

圈住"delay env 且这步没 done"：健康行走中（counter 本来 0，no-op）+ 正在窗口期内但 done 变 False（自己爬起来了）→ 清零死刑倒计时。

删掉这句的窗口缩水 bug：第 1 次摔躺 100 步爬起（counter=100）→ 第 2 次摔只剩 150 步窗口 → 第 3 次只剩 50 步 → 最终退化成普通 env。有了这句，同一 episode 每次摔倒都拿完整新窗口。

### 第 56 行：返回"篡改后"的 done 信号

```python
return self._truncated_buf | self._terminated_buf
```

第 39 行 `dones = super().compute()` 返回的是当时新算的张量（不是 buffer 引用），之后原地改 `_terminated_buf` 不会影响 `dones` 变量。若写 `return dones`，所有拦截白干——返回的仍是带终止信号的旧值。必须从两个 buffer 重新拼，拦截才体现在返回值里。

另外被注释掉的第 50 行 `# self._truncated_buf[not_ready] = False`：**超时从不被拦截**，delay env 躺地上 20 秒 episode 时钟照走，到点照样因 truncation reset——窗口只对"摔倒类终止"生效。

### 时间线串联四段逻辑

```
step 100: 摔倒 done=T → counter 0→1 → 拦截（not_ready 段）
step 101-179: 躺着 done=T → counter 涨到 80
step 180: 爬起来 done=F → 第 54 行赦免 counter→0，episode 继续活
step 500: 又摔 done=T → counter=1 → 又是完整 250 步新窗口
step 501-750: 没爬起来 → counter 到 250 → ready 段放行 reset + counter→0
step 751: 从 Recovery 倒地帧重生，done=T → counter=1 → 新窗口（第 51 行清零的功劳）
step 12000: episode 满 20s → truncated=T → 不拦截，正常超时 reset
```

---

## Q8: 终止的 -200 惩罚是怎么计算的？

**结论：-200 是权重不是实际扣分值。实际每步扣 -200 × 1 × dt = -200 × 0.02 = -4.0，且只在放行终止的那一步扣一次；窗口期内惩罚被冻结为 0；超时（truncation）不扣。**

### 1. 奖励函数本体：读"被篡改后"的 terminated buffer

mjlab 内置 `is_terminated` 就是读 `termination_manager.terminated` 属性转 float，而该属性就是 `_terminated_buf`。这正是与恢复窗口联动的关键：`DelayedTerminationManager.compute()` 在窗口内把 `_terminated_buf` 改 False → `is_terminated` 读到 0 → 惩罚冻结；只有计数器到 250 放行那一步 buffer 保持 True → 才产生惩罚。

### 2. RewardManager 合成公式：乘权重再乘 dt

```python
value = term_cfg.func(self._env, **term_cfg.params)
value = value * term_cfg.weight * scale      # scale = dt（scale_by_dt 默认 True）
self._reward_buf += value
```

调用处是 `reward_manager.compute(dt=self.step_dt)`，所以每项实际贡献 = 函数返回值 × weight × step_dt。

### 3. 代入本项目数值

`step_dt = timestep 0.005 × decimation 4 = 0.02s`，weight=-200：

| 场景 | 函数值 | 计算 | 实际奖励 |
|---|---|---|---|
| 正常活着 | 0 | -200 × 0 × 0.02 | 0 |
| 窗口期内躺着（被拦截） | 0（buffer 被改 False） | -200 × 0 × 0.02 | **0** |
| 窗口超时放行那一步 | 1 | -200 × 1 × 0.02 | **-4.0** |
| 普通 env 摔倒那一步 | 1 | 同上 | -4.0 |
| 20 秒超时（truncation） | 0 | — | **0** |

量纲对比：所有奖励同乘 dt，每步正奖励 ≤ 0.02，20 秒 episode（1000 步）理想回报上限约 20+。一次失败恢复 = -4，约等于丢掉 200 步（4 秒）满分行走收益。

### 4. 两个免责设计

- **超时免责**：`time_out` 是 `time_out=True` 终止项，写 `_truncated_buf`；`is_terminated` 只读 `_terminated_buf`——走满 20 秒被截断不扣分，只有真摔倒（bad_orientation 70°、bad_base_height 0.5m）才扣；
- **窗口冻结**：调用顺序保证生效——env step 里先 `termination_manager.compute()`（拦截已写进 buffer），后 `reward_manager.compute()`，reward 读到的就是篡改后的值。

### 5. 一次失败恢复的总账

delay env 摔倒后 5 秒没爬起来：**恰好一次 -4.0**（放行那一步），窗口内 249 步惩罚全 0。自救成功：0 惩罚 + episode 继续 accrue 正奖励。梯度信号清晰——"在 5 秒内站起来"的价值优势 ≈ 4 + 后续存活收益。

---

## Q9: 超时（truncation，活满最长运行时间）有没有算在惩罚里？

**结论：没有。`is_terminated` 只读 `_terminated_buf`，超时写 `_truncated_buf`，两者完全隔离——活满 20 秒是免费的。但超时通过 value bootstrapping 被"隐性惩罚"，不需要也不应该显式扣分。**

### 1. 代码层面：两条 buffer 完全隔离

- `time_out=True` 的终止项（`time_out`，20s）→ 写 `_truncated_buf`；
- `time_out=False` 的项（`bad_orientation` 70°、`bad_base_height` 0.5m、非法触地）→ 写 `_terminated_buf`；
- `is_terminated` 奖励只读 `_terminated_buf` → 活满 20 秒不扣分。

### 2. 超时被"隐性惩罚"——value bootstrapping

PPO/TD 学习里 terminal state 的 value 目标不同：

```
正常超时（truncation）：V(s_T) 目标 = r_T + γ·V(s_{T+1})   ← 时间到了，但明天还会继续
真终止（termination）：V(s_T) 目标 = r_T                    ← 世界毁灭，没有明天
```

rsl_rl 对超时 env 的 next_value 用 V(s_{T+1})（bootstrap），对真终止用 0。所以：

- 超时本身不扣分，但 value function 知道"活着有未来收益"；
- 摔倒（-4 + 无未来）vs 活满（0 + γV(未来)）的差距在 value 层面被正确区分；
- 若给超时也扣 -200 会双重惩罚（扣分 + value 低估），教策略"19 秒自杀比活到 20 秒更划算"——classic bug。

### 3. 为什么不惩罚超时是正确的：活着本身值钱

设计是让"活着"有正收益流：每步任务奖励 × dt 持续累积，`is_terminated` -200 只打真失败。超时的机会成本就是被截断的未来收益（γV），不需显式惩罚。与 Isaac Lab 官方 velocity 任务一致（`is_terminated` 惩罚 + `time_out` 不惩罚）。也没有用 legged_gym 时代的 alive_bonus——等价信息编码在"任务奖励 × 存活时长"里，活着 = 每步 ~0.02 收益，效果同 alive bonus 但更 task-aligned。

### 4. edge case：窗口超时 vs episode 超时的竞争

truncation 时钟是墙钟时间，躺着也在走。delay env 摔倒后一直爬不起来时，5 秒窗口通常先于 20s episode 超时触发 reset，所以"躺地白嫖超时"基本不存在。

但存在真实漏洞：若 env 在 t=16s 摔倒，窗口到 t=21s > 20s，则 20s truncation 先触发——**免费结束一次失败**（摔倒时间晚于 15s 且 4 秒内爬不起来的场景）。影响很小（value bootstrap 仍把后续 value 算进去，且频率低），但严格说是个漏洞。修法可以是窗口期结束时的超时也写 `_terminated_buf`，作者没这么做。

### 5. 时间线示例

```
t=0     episode 开始（从 Recovery 帧或正常帧）
t=X     摔倒 → 窗口开始（-200 惩罚冻结）
t=X+5s  仍没爬起 → reset（真终止，-4.0，恰好一次）
        → 新 episode 从 Recovery 帧开始 → 循环
t=20s   活到 20s（没摔或每次自救成功）→ truncation，免费，V 有未来
```

---

## Q10: metrics.py 的 mean_delay_steps 是计算什么的？

**结论：训练监控指标——"此刻平均每个 delay env 连续躺了多少步"，量化摔倒后爬不起来的程度；只进 TensorBoard 不进 reward。**

### 计算内容

```python
mean_delay_steps = sum(delay_counters) / sum(delay_env_mask)
```

- 分子：所有 delay env 的"连续 done 步数"计数器之和（健康=0，摔倒入窗口后 1→250）；
- 分母：delay env 总数（固定，如 4096×40% = 1638）；
- 分母是全体 delay env 而非"正在躺的"——健康 env 贡献 0 稀释均值。一半躺满 250、一半健康时指标 = 125。衡量的是整体时间占比意义上的"躺地负担"；
- `getattr(tm, ..., None)` 探测式访问：未启用延迟机制时（普通 TerminationManager 没这些属性）优雅返回 0 不炸。

### 数值怎么读

| 指标值 | 含义 |
|---|---|
| → 0 | delay env 基本健康行走（少摔或秒爬起） |
| 中等波动 | 一部分 env 在窗口期内挣扎 |
| 高位 | 大量 env 摔倒爬不起来，5 秒窗口躺满 |

训练早期高位；策略学会自救后（摔倒更少 / 窗口内快速爬起 counter 清零）陡降——README"约 2 万 iter 突然学会摔倒爬起、指标跳变"最直接的信号就是这个曲线跳水，对应 Q6 的"失败循环大规模瓦解"。

### 怎么进 TensorBoard

注册在 `amp_env_cfg.py` 的 metrics 里，与 `mean_action_acc`（动作二阶差分均值，越低越平滑）并列。每个 policy step 调 `metrics_manager.compute()` 累进 episode sums；env reset 时按步数平均，以 `Episode_Metrics/mean_delay_steps` 写进日志。**是 metric 不是 reward：不进 reward buffer、不影响梯度，纯监控仪表。**

---

## Q11: events.py 整个文件都干了什么？

**结论：它是 AMP 环境的"重置状态引擎"——一个单例类（数据加载/分流/写状态）+ 两个薄事件回调（startup 安装 + reset 采帧），是恢复机制的数据侧大本营。**

### 文件全景

```
MotionResetManager（单例）           两个事件回调（注册在 amp_env_cfg.py）
├─ get()          获取单例          ├─ init_motion_loader   → mode="startup"
├─ init()         加载动作帧库       │    加载数据 + 安装延迟终止管理器
├─ reset()        分流两类 env       └─ reset_from_motion_data → mode="reset"
├─ _write_reset_state()  写仿真状态       按身份分流采帧重置
└─ _concat_frames()      多段动作拼接
```

### 阶段一：startup —— init_motion_loader（训练只跑一次）

1. **喂数据**：单例 `init()` 把 WalkandRun（17 段 7,417 帧）和 Recovery（1 段 2,575 帧）读入、拼成两个大张量池常驻 GPU；
2. **动手术**：`torch.randperm` 抽 ratio 比例 env（G1 训练 0.4、play 1.0）打永久标记，偷换 `env.termination_manager` 为 `DelayedTerminationManager`。

细节：

- `num_delay > 0 and max_delay_steps > 0` 双防御——ratio 或步数配 0 则机制整体关闭（向后兼容）；
- 单例 + **motion_dir 做缓存键**：同进程第二个 env 用同样数据不重复加载，不同机器人/目录各存一份互不干扰（多任务场景设计）。

### 阶段二：reset 时 —— reset_from_motion_data（按身份分流）

- 普通 env → 从 WalkandRun 池随机采帧（开局行走/跑步姿态）；
- delay env → 从 Recovery 池采帧（开局大概率倒地姿态，逼练爬起），**未配 Recovery 目录时优雅回退** WalkandRun；
- `_get_delay_env_mask` 用 isinstance 探测——未安装延迟管理器时全体按普通 env 走，文件可独立于延迟机制使用。

### 阶段三：_write_reset_state（物理落地的三个工程细节）

1. **地形高度修正**（注释 "Key Fix for terrain"）：动作数据 z 是相对地面录的，Rough 地形上 env 原点 z 不同，需 `地形 z + 动作 z` 叠加——否则机器人被埋进土里或悬空；
2. **关节限位 clamp**：动捕数据可能略超软限位（重定向误差），clamp 避免约束求解器爆炸；
3. **写入顺序**：root pose → root velocity → joint state，一次性对齐完整物理状态，无位置/速度错位。

### 阶段四：_concat_frames（数据预处理）

把每段动作沿帧维度拼成一个大池，root 量取 body 0（pelvis）。好处："随机选段再选帧"简化为**一次 randint 均匀采全库**；代价是**无法按段加权**（对比 DroidUpE1 的 amp_motion_weights 逐段配权——这里时长长的段天然被采更多；AMP 判别器走另一个 loader 有自己的采样）。

### 两个回调为什么是薄委托

mjlab 事件签名是 `func(env, env_ids, **params)` 每 step 调用。逻辑收进单例、回调只转发：(1) 帧库只加载一次；(2) 状态可被其他模块引用。

### 本文件在恢复机制里的位置

```
init_motion_loader (startup, 本文件)      ← 数据加载 + 手术安装
        ↓
DelayedTerminationManager (terminations.py) ← 拦 done、开窗口（Q4/Q7）
        ↓
reset_from_motion_data (reset, 本文件)     ← 窗口超时后按身份采帧重置
        ↓
rewards.py 的 mask 逻辑                    ← 窗口期奖励切换（Q6）
```

本文件管"从哪来"（数据池）和"到哪去"（写仿真），"何时"（窗口计时）在 terminations.py，"值多少"（奖励）在 rewards.py。

---

## Q12: init() 开头 `if motion_dir in self.walk_run_frames: return` 是在干什么？

**结论：缓存/幂等检查——"这份数据加载过了就别再加载"，用路径做 key 防止重复 IO。**

### 逐行解释

- `self.walk_run_frames` 是 dict：key = `motion_dir`（数据目录路径），value = 拼接好的帧池张量；
- `motion_dir in self.walk_run_frames` = "该目录数据已加载过" → 直接 return，跳过整个 MotionLoader 读盘 + 拼接流程。

### 为什么需要

`MotionResetManager` 是跨 env 存活的单例（`_instance` 类变量，不随 env 销毁）。单个 env 的 startup 事件只触发一次，但同一进程里第二次创建 env 时检查会命中缓存。典型场景：

1. 训练 → 存档 → play：某些 workflow 同进程重建 env（如 eval 阶段新建 num_envs=1 的 env），不重新加载几十 MB NPZ；
2. 多任务注册：G1-AMP-Flat 和 G1-AMP-Rough 共用同一 motion_dir，后创建的 env 秒过 startup；
3. rsl_rl 导出 ONNX 时可能再实例化 env，同样命中。

缓存粒度是目录路径——不同机器人用不同目录会各自加载一份（按路径分桶），互不干扰，这也是选 dict 而非单成员变量的原因。

---

## Q13: events.py 逐行中文注释版

> 完整逐行注释见对话记录，此处收录精要版（文件全貌 + 关键行注释）。

### 文件结构

```python
# 导入区
from __future__ import annotations          # 类型注解兼容新语法
from typing import TYPE_CHECKING            # 仅类型检查时导入
import torch
from mjlab.entity import Entity             # 机器人实体（读写仿真状态）
from mjlab.managers.scene_entity_config import SceneEntityCfg  # 关节/刚体过滤器
if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv  # 环境类（仅注解用）
from ...ampmotion_loader import MotionLoader        # NPZ 读取器
from ...terminations import DelayedTerminationManager  # 延迟终止管理器

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")  # 默认：名为 robot 的实体、全部关节
```

### MotionResetManager 单例

```python
class MotionResetManager:
  _instance = None                    # 全局唯一实例的存放处

  def __init__(self):
    self.walk_run_frames = {}         # 缓存 {目录: 行走帧池}
    self.recovery_frames = {}         # 缓存 {目录: 恢复帧池}

  @classmethod
  def get(cls):
    if cls._instance is None:
      cls._instance = cls()           # 没有就建，有就复用
    return cls._instance
```

### init()：加载数据（带幂等守卫）

```python
def init(self, env, motion_dir, recovery_dir=None):
    if motion_dir in self.walk_run_frames:
        return                        # Q12：加载过就直接返回（缓存命中）

    loader = MotionLoader(motion_dir=motion_dir, ..., recovery_dir=recovery_dir)
    # 读 NPZ；tgt_body_indexes 等参数传占位值（reset 用不到 AMP 判别器那些）

    self.walk_run_frames[motion_dir] = self._concat_frames(loader.motion_data)
    # 多段动作拼成一个大池子，按路径存缓存
    # 打印：加载了几段、共几帧

    if loader.motion_data_recovery:   # 有恢复数据才存
        self.recovery_frames[motion_dir] = self._concat_frames(loader.motion_data_recovery)
```

### reset()：按身份分流采帧

```python
def reset(self, env, env_ids, motion_dir, asset_cfg):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs)   # None = 重置全部
    if len(env_ids) == 0:
        return                                  # 空集直接结束

    delay_mask = self._get_delay_env_mask(env)  # True/False 数组或 None
    if delay_mask is not None:                  # 启用了延迟机制
        is_delay = delay_mask[env_ids]          # 挑出本次env的标记
        delay_ids = env_ids[is_delay]           # 延迟env编号
        normal_ids = env_ids[~is_delay]         # 普通env编号
    else:                                       # 未启用
        delay_ids = env_ids[:0]                 # 延迟集合=空
        normal_ids = env_ids                    # 全按普通处理

    # 普通env → 行走池采帧
    self._write_reset_state(env, normal_ids, self.walk_run_frames[motion_dir], asset_cfg)

    # 延迟env → 恢复池采帧（没恢复数据就退回行走池）
    recovery = self.recovery_frames.get(motion_dir)
    frames = recovery if recovery is not None else self.walk_run_frames[motion_dir]
    self._write_reset_state(env, delay_ids, frames, asset_cfg)

def _get_delay_env_mask(self, env):
    tm = env.termination_manager
    if isinstance(tm, DelayedTerminationManager):  # 被偷换过=启用了
        return tm._delay_env_mask
    return None                                    # 没启用
```

### _write_reset_state()：写仿真状态

```python
def _write_reset_state(self, env, env_ids, frames, asset_cfg):
    idx = torch.randint(0, total_frames, (num_reset,), device=env.device)
    # 核心：每个env独立随机抽一帧

    # --- 根部位姿 ---
    root_pos = frames["root_pos"][idx]        # 抽中帧的位置
    root_quat = frames["root_quat"][idx]      # 抽中帧的朝向
    positions = env.scene.env_origins[env_ids].clone()  # env原点（复制不改原数组）

    terrain_z = positions[:, 2].clone()       # 备份地形高度
    positions[:, 2] = terrain_z + root_pos[:, 2]
    # 地形修正：新z = 地形高度 + 动作相对高度（否则埋土/悬空）

    root_pose = torch.cat([positions, root_quat], dim=-1)   # 7维位姿
    asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)  # 写入MuJoCo

    # --- 根部速度 ---
    root_vel = torch.cat([frames["root_lin_vel"][idx], frames["root_ang_vel"][idx]], dim=-1)
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
    # 带着动捕速度开始（比如摔倒时的下落速度）

    # --- 关节状态 ---
    joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos_clamped = joint_pos[:, asset_cfg.joint_ids].clamp_(
        joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # 关节角夹进软限位（动捕重定向可能微小超限，防物理引擎爆炸）

    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)  # list转张量
    asset.write_joint_state_to_sim(joint_pos_clamped, joint_vel[...], env_ids, joint_ids)
    # 至此位置+朝向+速度+关节全部对齐写入
```

### _concat_frames()：多段拼池

```python
for motion in motions:              # 每段=一个NPZ
    root_pos_list.append(motion["body_pos_w"][:, 0, :])   # body 0 = pelvis
    # ...其余同理
return {k: torch.cat(各list, dim=0)}  # 沿帧维拼接 → 一次randint均匀采全库
```

### 两个事件回调（薄转发层）

```python
def init_motion_loader(env, env_ids, motion_dir, recovery_dir=None,
                       delay_reset_env_ratio=0.0, max_delay_steps=0):
    MotionResetManager.get().init(env, motion_dir, recovery_dir)  # 加载（带缓存）

    num_delay = int(env.num_envs * delay_reset_env_ratio)  # 4096×0.4=1638
    if num_delay > 0 and max_delay_steps > 0:              # 任一为0=机制关闭
        delay_mask = torch.zeros(env.num_envs, dtype=torch.bool)
        delay_indices = torch.randperm(env.num_envs)[:num_delay]  # 随机抽编号
        delay_mask[delay_indices] = True                          # 打终身标记
        env.termination_manager = DelayedTerminationManager(      # 偷换手术
            base=env.termination_manager,                         # 继承原状态
            delay_env_mask=delay_mask, max_delay_steps=max_delay_steps)

def reset_from_motion_data(env, env_ids, motion_dir, asset_cfg=...):
    MotionResetManager.get().reset(env, env_ids, motion_dir, asset_cfg)  # 纯转发
```

### 一句话总结

启动时加载数据成大池+抽env打标+偷换终止管理器；每次重置时按标记分两拨、各自随机抽帧、把位姿/速度/关节一次性写进仿真（z叠加地形、关节夹限位）。

---

## Q14: rewards.py 的 reward 是怎么计算的？

**结论：文件定义单项公式，完整 reward = Σ(函数值 × weight × dt)（Q8 的合成公式）。文件分三层：mask 工具函数（窗口期奖励切换）→ 9 个生效奖励项 → 状态相关的总值。**

### 第一层：三个工具函数——延迟窗口的"奖励开关"

```python
def _get_delay_env_mask(env):
    # 关键：delay_env_mask & (delay_counters > 0)
    # 不是"所有delay env"，而是"delay env 且正躺在恢复窗口内"（counter>0）
    # 健康行走的 delay env 不激活

def _apply_delay_env_reward_scaling(env, reward, mask_delay, ratio):
    # 缩放版：躺地env拿 ratio 倍，其他env原值
    # 用于"归零/调低"任务奖励（速度跟踪类 ratio=0.0）

def _apply_delay_env_reward_mask_only(env, reward, mask_delay, ratio):
    # 独占版：只有躺地env有值，其他env一律0
    # 用于 track_root_height（站起来专属奖励，普通env走路时不被"够不够高"干扰）
```

### 第二层：9 个奖励项

**任务奖励（正）**

| 项 | weight | std | 公式 | 说明 |
|---|---|---|---|---|
| track_anchor_linear_velocity | 1.0 | 1.0 | `exp(-‖v_cmd_world - v_actual‖²/std²)` | 指令xy经yaw旋转到世界系，z置0；误差0→1分，1m/s→0.37分 |
| track_anchor_angular_velocity | 1.0 | 3.14 | `exp(-(ωz误差+ωxy²)/std²)` | yaw角速度跟指令 + roll/pitch角速度抑制 |
| track_root_height | 1.0 | 0.3 | `exp(-((h_default-h_cur)/std)²)` | **mask_only + ratio=3.5**；躺着(0.2m)≈0.018分，起身到0.75m≈0.98分，跳变~50倍 |

**存活/终止**

| 项 | weight | 说明 |
|---|---|---|
| is_terminated | -200 | 只在真终止放行那步 -4.0；窗口冻结、超时免责（Q8） |

**正则化惩罚（持续小额）**

| 项 | weight | 说明 |
|---|---|---|
| joint_acc_l2 | -2.5e-7 | 关节加速度平方和（平滑） |
| joint_pos_limits | -10.0 | 超软限位比例（保护电机） |
| action_rate_l2 | -0.01 | 相邻动作差平方（防抖） |

**接触相关（负）**

| 项 | weight | 说明 |
|---|---|---|
| feet_slip | -0.25 | 触地脚水平滑速²求和；仅在速度指令激活时罚；sensor配history_length=4逐子步检查 |
| self_collisions | -0.1 | 自碰撞力>10N的**子步计数**（碰撞可能子步间发生又弹开，瞬时值漏检，用history缓冲） |

soft_landing 在配置里被注释掉，未生效。

### 第三层：不同状态每步实际 reward（dt=0.02）

| 状态 | 构成 | 每步（×dt后） |
|---|---|---|
| 健康行走跟得好 | 速度1.0×~0.9 + 角速1.0×~0.8 + 0.5×~0.9 + 正则~-0.3 | **≈ +0.025** |
| delay env躺窗口内 | 速度/角速奖励0（ratio=0）、height 3.5×0.02、终止罚0（冻结） | **≈0 + 微弱正则罚**，起身时height单调涨 |
| 摔倒放行那步 | is_terminated -200×1 | **-4.0** |

本质：健康时奖励引导"跟指令走"，躺地时自动切换成"站起来"（height×3.5独占激活），两种模式共享同一 reward 接口，靠三个 mask 工具函数无痕切换。

---

## Q15: 自碰撞检测能定位"哪两个关节互相碰撞"吗？

**结论：当前配置不能——subtree(pelvis)=整树聚合 + num_slots=1，只能输出"有无/次数"标量。但改配置可以到"部位 vs 子树"粒度，精确到 geom pair 需绕过传感器直接读接触数组。**

### 1. 当前配置为什么定位不了

G1 的 self_collision 传感器：`primary=ContactMatch(mode="subtree", pattern="pelvis")` + 同样 secondary。pelvis 是 G1 整树根，**subtree(骨盆) = 整个机器人**——MuJoCo subtree contact sensor 天然把子树内所有 geom pair 接触**汇总**成一个数（found=接触对数，force=归约代表值）。加上 `num_slots=1`（每 primary 只留 1 个代表接触），输出 `[B, 1, H, 3]`，装不下 23 个部位的区分信息。reward 拿到的 `hit.sum(-1)` 只是"这步内多少子步发生了力>10N 的自碰"，不知道是大腿磕腹部还是手臂撞躯干。

### 2. MuJoCo 机制支持定位，三种方案

**方案 A（推荐）：逐部位 primary + 整树 secondary**

```python
primary=ContactMatch(
    mode="body", pattern=r".*_collision", entity="robot",
    exclude=(r"^(left|right)_foot[1-7]_collision$",),  # 排除脚（碰地误报）
),
secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
```

输出变 `[B, P, H, 3]`（P=部位数），`sensor.primary_names[i]` 给槽位→部位名映射——可回答"部位 X 碰了子树里某个 geom"，配合排除法基本能推断对手。缺点：MuJoCo 原生 sensor 不直接给对端 geom 名。

**方案 B（精确到 geom pair）：直接读 data.contact**

```python
for i in range(ncon):
    name1, name2 = model.geom(data.contact.geom1[i]).name, model.geom(data.contact.geom2[i]).name
    # 两个都在 *_collision 集合 → 自碰 pair (name1, name2)
```

最精确但每步遍历接触数组，4096 env 下开销不小，且 mjlab ContactSensor 抽象未直接暴露此接口。

**方案 C（奖励工程视角）：通常不需要定位**

penalty 目的是整体抑制自碰，-0.1 小权重让策略自然避开所有自碰姿势。想知道"哪两个关节老打架"应在分析阶段用 A/B 跑 play 收集统计，而非塞进训练 reward。

### 3. G1 自碰碰撞体配置

机器人用 FULL_COLLISION（23 个 *_collision geom 全开自碰）：pelvis、hip×2、thigh×2、shin×2、foot×14、hand×2、head、肩×2、肘×2 等。自碰体 condim=1（仅法向力无摩擦）——这就是阈值 10N 而非更小的原因：condim=1 碰撞解析较"软"，微小穿透力不大，10N 过滤真实硬碰。

### 总结表

| 问题 | 答案 |
|---|---|
| 现在能定位哪两个部位碰了吗 | ❌ subtree聚合+num_slots=1，只有"有无/次数" |
| MuJoCo 支持吗 | ✅ 逐部位 primary + primary_names 映射到"部位 vs 子树" |
| 精确 geom pair | 读 data.contact，有性能代价 |
| 训练需要吗 | 一般不需要；定位用于离线分析 |

---

## Q16: 健康行走为什么有三个正奖励？是不是重复计算？

**结论：不是重复——是"按指令行走"任务拆成的三个子目标（平移速度 / 转向+躯干稳 / 骨盆稳），各测不同物理量、各有独立 exp 核。**

### 三项对照

| 项 | 管什么 | 公式 | 跟得好 |
|---|---|---|---|
| track_anchor_linear_velocity (1.0) | 速度对不对：走多快往哪走 | exp(-‖v_cmd−v_actual‖²/1.0²) | 0.9 |
| track_anchor_angular_velocity (1.0) | 转向对不对 + torso 稳 | exp(-((ωz−ωz_cmd)²+‖ω_xy‖²)/3.14²) | 0.8 |
| body_ang_vel_xy_l2 (0.5) | 骨盆别晃（roll/pitch→0） | exp(-‖ω_xy_pelvis‖²/3.14²) | 0.9 |

### 为什么这么拆（行走 reward 工程惯例）

1. **任务天然分解**：平移速度、转向速率、躯干稳定是独立子问题。可以"速度对但晃"（第1高、2/3低）或"稳但没跟速度"（2/3高、1低）——分开给分，每个子目标都有独立稠密的梯度；
2. **exp 核有界 (0,1]**，weight 即相对重要性：速度 1.0、转向 1.0、稳定 0.5，完美行走任务分上限 2.5/步（×dt=0.05）；
3. **部分学分**：早期速度跟不动时"站稳"仍能拿分——先学站稳再学走对，避免奖励稀疏；
4. **可监控**：TB 里逐项画曲线，"速度跟上了但躯干晃"一眼可见。

### 两个易混点

- **第2、3项都在管 roll/pitch，半重复是刻意**：锚点不同（第2项 torso_link 且混着 yaw 跟踪，第3项纯 pelvis），躯干稳定对人形步态重要，双倍强调（有效 1.0+0.5）；
- **`body_ang_vel_xy_l2` 名字带 _l2 却给正分**：命名沿袭 Isaac Lab 惩罚习惯，实现是 exp(-err/std²) 核——误差小→接近1分，本质"奖励不晃"；
- **没有 track_root_height**：mask_only，普通 env 健康行走拿 0，只有 delay env 躺窗口内激活（Q14）。

---

## Q17: Flat 地形和 Rough 地形上的碰撞数量有什么不同？

**结论：实际接触数量 rough > flat（mesh 多接触对 vs 单平面），且代码里对两个任务显式配置了不同的碰撞容量预算——flat 全面调低省显存，rough 调高防接触爆容量。**

### 配置对比（env_cfgs.py）

flat 版从 rough 版继承后第一件事就是调低这四个数：

| 参数 | Rough | Flat | 含义 |
|---|---|---|---|
| nconmax | 48（硬上限） | None（启发式） | 每 world 最多同时存在多少接触对 |
| njmax | 1500（继承） | 640 | 每 world 约束行数上限（自碰 condim=1、脚 condim=3） |
| ccd_iterations | 500 | 50 | 连续碰撞检测迭代（防穿透） |
| contact_sensor_maxmatch | 500 | 256 | contact sensor 匹配容量 |

### 物理直觉

1. **Rough 接触对更多更碎**：粗地形是 mesh，一只脚（7 个 capsule）踩起伏面可能同时接触多个三角面，叠加自碰、踹台阶棱等瞬时接触。nconmax=48 显式留余量（基础 35 不够）；CCD 500 是"stability 但避免 EPA 缓冲 OOM"的调优平衡点；
2. **Flat 接触对少而稳定**：平地上每脚就是 capsule vs 无限平面，数量可预测。nconmax=None 自动算，njmax 压到 640 够用。**核心动机是省显存**：数组按 num_envs × 容量 批量分配，4096 env 时容量直接决定 GPU 占用；
3. **检测逻辑相同**：同一套 contact sensor、同一个 self_collision_cost reward，差异只在容量预算。nconmax 超容量的接触会被 MuJoCo Warp 丢弃——rough 给足 48 防高速落脚/摔倒时物理失真，flat 摔倒也就 20 来个接触对，自动启发式够。

### flat 的其他差异

- 地形换 plane、删 terrain_generator；
- 删 raycast 传感器 + height_scan 观测（平地没得扫）→ **actor 观测维度比 rough 少 187 维，两个任务 checkpoint 不通用**；
- 删地形 curriculum（play 模式扩大指令范围）。

---

## Q18: play.py 报 AttributeError: 'PlayConfig' object has no attribute 'wandb_run_path'

**结论：play.py 从 mjlab 上游借来时删 wandb 功能删漏了——PlayConfig 没定义 wandb_run_path/registry_name 字段但代码在用。已修复：补上两个字段定义。**

### 原因

- PlayConfig 定义的字段里没有 `wandb_run_path`（第 176 行的 `registry_name` 同类幽灵引用）；
- 不带 `--checkpoint-file` 运行时走到 `cfg.wandb_run_path` 直接 AttributeError；
- 原意是抛"请提供 checkpoint 或 wandb 路径"的友好 ValueError，报错代码自己先炸了。

### 修复（已应用）

给 PlayConfig 补上：

```python
wandb_run_path: str | None = None
"""Optional WandB run path to download a checkpoint from when checkpoint_file is not set."""
registry_name: str | None = None
"""Optional motion registry name for tracking tasks in dummy mode."""
```

### 正确用法

trained 模式必须加载 checkpoint：

```bash
python scripts/play.py Unitree-G1-AMP-Flat \
  --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run_dir>/model_<iter>.pt
```

带 --checkpoint-file 不会碰 wandb 分支。另注意：训练启动初期（如 06:47 启动的 run）还没存出任何 model_*.pt，此期间 play 无 checkpoint 可载。

---

## Q19: play 的时候有几个机器人？是 40% 躺地上吗？

**结论：默认 1 个机器人；躺地相关 delay 比例不是 40% 而是 100%——play 模式专门提到 1.0，用于展示恢复能力。**

### 数量与比例

- 基础配置 `num_envs=1`（amp_env_cfg.py），play.py 只有显式传 `--num-envs N` 才覆盖；
- env_cfgs.py 的 play 分支：`delay_reset_env_ratio = 1.0` + `episode_length_s = 1e9`；
- 效果链：reset 从 Recovery 数据（fallAndGetUp 2,575 帧"摔倒-爬起"全程）随机采帧——大量帧是躺地/半躺姿态，**开局经常直接躺地上再爬起来**；摔倒后完整 5 秒窗口自救；无限 episode 不摔永不重置。

### play vs train 对比

| | train | play |
|---|---|---|
| 机器人数量 | 4096 | 1（--num-envs 可改） |
| delay env 比例 | 0.4 | 1.0 |
| reset 帧来源 | Recovery 池（40% 部分） | Recovery 池 |
| episode 时长 | 20s | 1e9（无限） |
| 推扰/噪声 | 有 | 无 |

改机器人数量：`python scripts/play.py Unitree-G1-AMP-Flat --num-envs 10 --checkpoint-file ...`

---

## Q20: 不想每次提供模型地址，希望 play 自动加载最新 checkpoint

**结论：已改 play.py——checkpoint 加载改为三级优先：--checkpoint-file > --wandb-run-path > 自动扫描。**

### 自动扫描逻辑（新增分支）

都不传时：在 `logs/rsl_rl/<experiment_name>/` 下找**最新时间戳 run 目录**（目录名字典序即时间序），取其中**迭代数最大**的 `model_*.pt`，打印 `[INFO]: Auto-selected checkpoint: model_N.pt (run: <dir>)`；没有任何 checkpoint 时抛出带指引的 FileNotFoundError。

### 用法

```bash
python scripts/play.py Unitree-G1-AMP-Flat --num-envs 10
```

自动跟随训练进度——训练存出更大的迭代 checkpoint 后，下次 play 自动选新的。

---

## Q21: 10 个机器人里为什么只有部分脑袋上有箭头？

**结论：箭头是速度指令的 debug 可视化，默认只画在"当前追踪的 env"上；按键 A（Show all envs）可让所有机器人都显示。**

### 箭头是什么

G1 配置 `viz.z_offset = 1.15` 把四类箭头画在头顶上方，每种颜色一个含义（velocity_command.py 的 `_debug_vis_impl`）：

| 颜色 | 含义 |
|---|---|
| 蓝 | 指令线速度（往哪走多快） |
| 绿 | 指令角速度（转向，z 轴） |
| 青 | 实际线速度 |
| 浅绿 | 实际角速度 |

指令箭头和实际箭头重合得越好 = 跟踪越好，这是肉眼评估策略的快捷方式。

### 为什么只有部分机器人有

debug 可视化默认只画**当前追踪的 env**（`get_env_indices` 返回 `[env_idx]`，`show_all_envs=False` 是初始值）。native viewer 的按键绑定（viewer.py）：

| 键 | 作用 |
|---|---|
| **A** | 切换 Show all envs（开→所有机器人画箭头） |
| **R** | 开/关 debug 可视化 |
| **, / .** | 上一个/下一个 env（切换追踪对象） |
| Enter | 重置 |
| 空格 | 暂停 |
| - / = | 减速/加速 |
| P | reward 曲线图 |

按 **A** 即可让 10 个机器人全部显示箭头。用 viser viewer 的话对应 GUI 里的 "Debug Viz → All envs" 复选框。

---

## Q22: AMP_mjlab 机器人的 action scale 是多少？

**结论：不是一个数，是逐关节 dict——`scale = 0.25 × effort_limit / stiffness`，物理含义"action=1 的偏移角恰好需要 25% 最大力矩"。**

### 公式与使用

g1_constants.py 按执行器组算 `G1_ACTION_SCALE[n] = 0.25 * e / s`；env_cfgs.py 灌进 `joint_pos_action.scale`。实际目标角 = 默认角 + action × scale。

stiffness 不手拍：`ARMATURE × (10Hz×2π)²`（10Hz 自然频率、阻尼比 2.0），ARMATURE 由两级行星齿轮折算（reflected_inertia_from_two_stage_planetary：`r1*G²+r2*g²+r3` 结构）。

### 数值（29dof G1，训练日志 out_features=29 印证）

| 关节组 | 电机 | effort (N·m) | stiffness | scale (rad) |
|---|---|---|---|---|
| 髋俯仰/侧滚、膝 ×6 | 7520-22 | 139.0 | ≈99.1 | **≈0.35** |
| 髋偏航×2、腰偏航 | 7520-14 | 88.0 | ≈40.0 | **≈0.55** |
| 腰俯仰/侧滚、踝 ×6 | 2×5020 | 50.0 | ≈28.4 | **≈0.44** |
| 肩/肘/腕侧滚 ×10 | 5020 | 25.0 | ≈14.2 | **≈0.44** |
| 腕俯仰/偏航 ×4 | 5010-16 | 10.0 | ≈8.6 | **≈0.29** |

action 归一化到 [-1,1] 后各关节"可用力矩百分比"对齐（都是 25%），不同关节偏移幅度不同。

### 对比

DroidUpE1 用单一固定 action_scale；AMP_mjlab 逐关节字典——G1 的 29 关节横跨 5 种电机（10~139 N·m 力矩差 10 倍+），单一 scale 无法兼顾。

---

## Q23: amp_env_cfg.py 里 action 的 scale=0.25 又是怎么回事？和 G1_ACTION_SCALE 什么关系？

**结论：0.25 是任务模板的通用默认值（注释明写 "Override per-robot"），G1 任务注册时被 env_cfgs.py 整体替换成 G1_ACTION_SCALE dict——G1 上最终生效的是 dict，0.25 用不到。**

### 两级配置流水线

```
第1层 任务模板 amp_env_cfg.py:
  JointPositionActionCfg(scale=0.25)   # 通用保守默认，任何机器人保底能跑
        ↓ 被覆盖
第2层 机器人覆盖 config/g1/env_cfgs.py:
  joint_pos_action.scale = G1_ACTION_SCALE   # G1 专属逐关节 dict（0.29~0.55 rad）
        ↓
运行时 mjlab BaseActionCfg:
  scale: float | dict[str, float]   # 两种类型都接受，float 广播、dict 按名查
```

### 为什么这么设计

manager-based RL 框架的标准分层（Isaac Lab 同款）：任务逻辑（奖励/观测/终止）写在模板与机器人无关；机器人参数（电机、scale、传感器匹配）注册具体任务时覆盖。模板 0.25 保证新机器人忘配 scale 也能安全幅度运行；执行器参数齐全的机器人用精确逐关节值替换。

另：`use_default_offset=True` 是另一半逻辑——目标角 = 默认关节角 + action × scale，动作空间围绕站立姿态对称。

---

## Q24: amp_env_cfg.py 里 body_pos_b / body_ori_b / body_lin_vel_b / body_ang_vel_b 这几个观测是什么？

**结论：一套"身体状态"观测函数（局部系，后缀 _b）——critic 用前 2 项做特权信息，AMP 判别器用全部 4 项与专家动作对比。**

### 四个函数

| 函数 | 算什么 | 坐标系 | 维度/身体 |
|---|---|---|---|
| robot_body_pos_b | 各部位相对锚点的位置 | 锚点（torso_link）系 | 3 |
| robot_body_ori_b | 各部位相对锚点姿态，旋转矩阵取前 2 行（6D 表示） | 锚点系 | 6 |
| robot_body_lin_vel_b | 各部位线速度 | 各 body 自己的体系 | 3 |
| robot_body_ang_vel_b | 各部位角速度 | 各 body 自己的体系 | 3 |

细节：ori 用 6D 旋转表示（matrix_from_quat 后取 [..., :2]）——比四元数学习友好（连续、无 q/-q 双重表示）；pos 相对锚点但速度在各自体系（"沿自己前方移动"与否，与全身朝向无关）。

### 跟踪的身体与锚点（G1 覆盖后）

模板 body_names=() 是占位（同 action scale 的"模板→覆盖"套路）；G1 填 13 个身体（pelvis + 双腿各 3 + 双臂各 3），锚点 torso_link。

### 各组用途与维度

| 组 | 项 | 维度 | 用途 |
|---|---|---|---|
| critic | pos_b + ori_b | 13×3+13×6=117 | 特权信息（部署时 critic 不上线） |
| amp | 全部 4 项 | 39+78+39+39=195 | AMP 判别器状态 |

验证：critic in_features=864 = (actor 单帧 96 + base_lin_vel 3 + 39 + 78) × history 4。

### 为什么恰好这 4 项

AMP 论文标准判别器状态：身体相对位置+姿态+线速度+角速度，完整刻画姿态-速度快照。且与 ampmotion_loader 专家 NPZ 数据逐项对齐（同坐标系约定），机器人观测和专家观测在同一空间对比，判别器才能公平打分。

---

## Q25: metrics 里的 mean_action_acc 是在哪里定义的？

**结论：mjlab 库内置指标（不是本仓库写的），定义在 site-packages/mjlab/envs/mdp/metrics.py，经多层 import * 进入 mdp 命名空间。**

### import 链

```
amp_env_cfg.py:
  L26  from mjlab.tasks.velocity import mdp        # 第一次导入
  L33  import src.tasks.amp_loco.mdp as mdp        # 第二次导入，覆盖上面的名字 ← 生效的是这个
       ↓
src/tasks/amp_loco/mdp/__init__.py:
  from mjlab.envs.mdp import *                     # 把 mjlab 库的 mdp 全部混入
       ↓
/home/zju/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/mjlab/envs/mdp/metrics.py
  def mean_action_acc(...)                          # 真正的定义
```

### 函数内容

动作离散二阶差分 a_t − 2a_{t-1} + a_{t-2}（加速度）取绝对值、对关节求均值——"动作有多抖"的量化，越低越平滑。TB 曲线名 Episode_Metrics/mean_action_acc。

### mdp 命名空间三个来源（查定义的口诀）

| 来源 | 例子 |
|---|---|
| mjlab 库内置 | mean_action_acc、is_terminated、joint_acc_l2、joint_pos_limits、action_rate_l2、bad_orientation、time_out |
| 本项目 amp_loco | mean_delay_steps、init_motion_loader、reset_from_motion_data、self_collision_cost、track_anchor_*、robot_body_pos_b 等 |
| velocity 任务包（env_cfgs.py 用） | UniformVelocityCommandCfg、terrain_levels_vel |

先查 src/tasks/amp_loco/mdp/，没有就去 site-packages/mjlab/envs/mdp/。

---

## Q26: 40% 环境站不起来时，episode length 是不是只有 250？总体平均应该多大？

**结论：250 是恢复窗口不是 episode 上限（上限人人 1000 步）。最坏情形 delay env 退化到 ≈250 步；"40% 全躺"的稳态下总体平均 ≈700；实测 838.49 说明大部分 delay env 已能走路。**

### 单 env 公式

delay env 第 t 步摔倒且爬不起来：E_delay = min(t+250, 1000)

| 情形 | episode 长度 |
|---|---|
| reset 采到躺地帧、爬不起来（t≈0） | ≈250 |
| 走 t 步后摔、爬不起来 | t+250（>1000 被 timeout 截断，即 Q9 白嫖超时） |
| 不摔/自救成功 | 1000 |

普通 env：E_normal = min(t, 1000)，无缓刑期。

### 总体平均 = 0.6·E_normal + 0.4·E_delay

| 场景 | E_normal | E_delay | 总体 |
|---|---|---|---|
| 最早期全员秒摔 | ~0 | ~250 | ≈100 |
| 正常 env 完美、delay 全失败 | 1000 | 250 | **=700** |
| 全完美 | 1000 | 1000 | =1000 |

### 实测验证（2026-09-23 训练 ~2h 的日志）

Mean episode length: 838.49，mean_delay_steps: 25.6。

- 0.6×950 + 0.4×E_d = 838.49 → E_d ≈ 671（delay env 平均活 ~670 步，远超 250）；
- mean_delay_steps=25.6：躺地期间 counter 锯齿均值 ~125，全体 delay env 时间平均 25.6 → 躺地时间占比 ≈20%；
- 交叉印证：大部分 delay env 已能走路（~80% 时间站着），少数在失败循环；若全卡死，均值会掉到 700、mean_delay_steps≈125，都没发生。

### 有用的不等式

同摔倒时刻下 E_delay ≥ E_normal + 250（缓刑期白送）→ delay env 经验里"倒地状态转移"采样密度天然比正常 env 高 4 倍+，正是机制设计目的。

---

## Q27: commands 里 lin_vel_x 上限写 3.0，curriculum 又写 2.0，哪个生效？

**结论：训练时 curriculum 覆盖 commands.ranges，真正生效的是课程阶段值；名义 3.0 在训练中采不到。play 清掉 curriculum 后才用满 3.0。**

### 两处分别是什么

```python
# commands（amp_env_cfg.py ~211）：名义采样范围
ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.5, 3.0), ...)

# curriculum.command_vel（~389-390）：按训练步数改写 ranges
{"step": 0,          "lin_vel_x": (-0.5, 1.0), ...}   # 开局
{"step": 5000 * 24,  "lin_vel_x": (-1.0, 2.0), ...}   # 扩到 2.0 为止
```

`commands_vel` 实现（curriculums.py）直接改 `cfg.ranges.lin_vel_x = stage[...]`——**原地覆盖**，不是取 min/max。

### 时间线（num_steps_per_env=24）

| common_step_counter | 学习 iter（约） | 实际 lin_vel_x |
|---|---|---|
| 初始化瞬间 | 0 | (-1.5, 3.0) 名义值，马上被覆盖 |
| > 0 | 启动后 | **(-0.5, 1.0)** |
| > 120000（=5000×24） | ≈5000 | **(-1.0, 2.0)**，之后不再扩 |

### 和 play 的差异

- train Flat：只 pop 掉 `terrain_levels`，`command_vel` 仍在 → 训练上限 **2.0**；
- play：`cfg.curriculum = {}`，并显式 `twist_cmd.ranges.lin_vel_x = (-1.5, 3.0)` → play 可以指令到 **3.0**。

这是 train/play 不一致：play 能下 3.0 的指令，但策略训练时最高只见过 2.0。若要对齐，要么把 curriculum 终段改成 `(-1.5, 3.0)`，要么把 commands/play 的上限改成 2.0。

---

## Q28: 这个项目到底启用了速度课程，还是直接用 3.0？

**结论：训练启用了速度课程，不是直接 3.0。你这次 Flat 训练的 env.yaml 里能直接看到 `curriculum.command_vel`。**

### 证据（你自己的 run）

`logs/rsl_rl/g1_amp_locomotion/2026-09-23_06-47-33/params/env.yaml`：

- `commands.twist.ranges.lin_vel_x: (-1.5, 3.0)` —— 名义默认值仍写着；
- `curriculum.command_vel` 存在，两阶段：step 0 → (-0.5, 1.0)，step 120000 → (-1.0, 2.0)；
- Flat 只 `pop` 了 `terrain_levels`，**没删** `command_vel`。

### 实际训练采样范围

| 阶段 | 条件 | 实际 lin_vel_x |
|---|---|---|
| 开局～约 5000 iter | step ≤ 120000 | **(-0.5, 1.0)** |
| 约 5000 iter 之后 | step > 120000 | **(-1.0, 2.0)** |
| 全程不会到 | — | **3.0** |

名义 3.0 只在 curriculum 尚未覆盖前的瞬间存在，第一步 curriculum 更新后就会被改成 1.0。

### 对照表

| 模式 | 地形课程 | 速度课程 | 实际前进速度上限 |
|---|---|---|---|
| train Rough | 有 | 有 | **2.0** |
| train Flat（你在跑的） | 无 | **有** | **2.0** |
| play | 全清 | 全清 | **3.0**（显式改回） |

---

## Q29: env_cfgs 里 motion_dir / recovery_dir 两套数据分别怎么采样？

**结论：reset 时按机器人身份分池采样——普通 env 只从 WalkandRun 抽帧，delay env 只从 Recovery 抽帧；AMP 判别器则把两池 clip 拼在一起按"先选段再选帧"采样。**

### 配置在干什么（你标的 111-114 行）

```python
cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir      # WalkandRun/
cfg.events["init_motion_loader"].params["recovery_dir"] = _recovery_dir  # Recovery/
cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir      # 仅作缓存 key
```

startup 时把两目录加载成两个帧池（以 motion_dir 路径为 key 存进单例）；每次 reset 用这个 key 取出对应池。

### 路径 A：环境 reset 采样（events.py）

```
要重置的 env_ids
    ├─ 普通 env（60%，~delay_mask）→ walk_run_frames 池
    └─ delay env（40%，delay_mask）→ recovery_frames 池
         （没有 Recovery 数据时回退 WalkandRun）
```

两边共用 `_write_reset_state`：

```python
idx = torch.randint(0, total_frames, (num_reset,))  # 每个 env 独立均匀抽一帧
# 写入 root pose/vel + joint pos/vel（z 叠加地形、关节 clamp）
```

采样细节：

| | WalkandRun（普通 env） | Recovery（delay env） |
|---|---|---|
| 数据 | 17 段拼成大池，共 7,417 帧 | 1 段 fallAndGetUp，2,575 帧 |
| 抽法 | 一次性 randint 均匀抽全库帧 | 同左 |
| 权重 | **按帧均匀**（长片段天然被采更多） | 单段内均匀 |
| 写入内容 | root 位姿/速度 + 关节角/角速度 | 同左（常是倒地/半倒姿态） |

注意：reset **不按片段加权**，是把所有片段 `torch.cat` 后按帧抽——walk_forward 长的段比 idle_turn 短的段更容易被采到。

### 路径 B：AMP 判别器采样（ampmotion_loader.py，另一条线）

`sample_random_frames` 把 `motion_data + motion_data_recovery`（18 个 clip）拼在一起：

```python
all_motions = self.motion_data + self.motion_data_recovery  # 17+1=18 段
motion_indices = torch.randint(0, len(all_motions), (num_samples,))  # 先均匀选段
frame_idx = torch.randint(0, num_frames, (1,))                       # 再在段内均匀选帧
```

和 reset 的区别：

| | reset（MotionResetManager） | AMP 判别器 |
|---|---|---|
| 两池关系 | **分开**：身份决定进哪个池 | **合并**：18 段一起抽 |
| 均匀性 | 按帧均匀（长段权重大） | 按段均匀（每段 1/18，短段相对更常被抽） |
| 用途 | 给机器人设初始物理状态 | 给判别器喂专家状态对比 |

### 一句话

你标的这两行是在告诉系统"走路数据在哪、爬起数据在哪"；真正采样时——**reset 按 40/60 身份分池各抽一帧，AMP 把两池 clip 合并后先选段再选帧**。

---

## Q30: 为什么 reset_from_motion 只传 motion_dir，不传 recovery_dir？

**结论：倒地数据在 startup 时已经用 motion_dir 当 key 存进 recovery_frames；reset 只要同一个 key 就能取出两池，不需要再传倒地目录。**

### 存的时候

```python
# init（startup，只跑一次）
self.walk_run_frames[motion_dir] = ...   # key = WalkandRun 路径
self.recovery_frames[motion_dir] = ...   # key 也是 WalkandRun 路径，不是 recovery_dir！
```

`recovery_dir` 只在 init 读盘时用一次，读完就绑在 `motion_dir` 这个主键下。

### 取的时候

```python
# reset（每回合）
self.walk_run_frames[motion_dir]                    # 普通 env
self.recovery_frames.get(motion_dir)                # delay env，同一个 key
# get 返回 None → 回退 WalkandRun
```

所以 `reset_from_motion.params["motion_dir"] = _motion_dir` 的含义是：**查哪一对帧池**，不是"只采走路数据"。

### 为什么这样设计

1. 一对一绑定：一套 WalkandRun 对应一套 Recovery，用走路目录路径当主键够用；
2. reset 事件签名简单（一个路径参数）；
3. 没配 Recovery 时优雅回退。

---

## Q19: play 的时候有几个机器人？是 40% 躺地上吗？

**结论：默认 1 个机器人；躺地相关的 delay 比例不是 40% 而是 100%——play 模式专门把它提到 1.0，用于展示恢复能力。**

### 数量：默认 1 个

基础配置 `num_envs=1`（amp_env_cfg.py），play.py 只有显式传 `--num-envs N` 才覆盖。

### 躺地比例：play 覆盖为 1.0

env_cfgs.py 的 play 分支：`delay_reset_env_ratio = 1.0` + `episode_length_s = 1e9`。效果链：

- reset 从 Recovery 数据（fallAndGetUp 2,575 帧"摔倒-爬起"全程）随机采帧——大量帧是躺地/半躺姿态，**开局经常直接躺地上再爬起来**；
- 摔倒后完整 5 秒窗口自救（max_delay_steps=250 不变）；
- 无限 episode：不摔永不重置。

### play vs train 对比

| | train | play |
|---|---|---|
| 机器人数量 | 4096 | 1 |
| delay env 比例 | 0.4 | 1.0 |
| reset 帧来源 | Recovery 池（40% 部分） | Recovery 池 |
| episode 时长 | 20s | 1e9（无限） |
| 推扰/噪声 | 有 | 无 |

注：500 iter 的 checkpoint 策略还很弱（README：约 2 万 iter 才突然学会摔倒爬起），play 时躺地后爬不起来属正常。

---

## Q30b: reset 只传 motion_dir 的完整调用链（带行号）

**按执行顺序把“倒地目录在哪被读、存在哪、reset 怎么取回来”钉死。**

### 1. 配置：倒地目录只塞给 init，不塞给 reset

`src/tasks/amp_loco/config/g1/env_cfgs.py` 109-114：

```
_motion_dir   = .../WalkandRun
_recovery_dir = .../Recovery

init_motion_loader.params["motion_dir"]   = _motion_dir     # 有
init_motion_loader.params["recovery_dir"] = _recovery_dir   # 倒地目录在这里传入
reset_from_motion.params["motion_dir"]    = _motion_dir     # 只有这个
```

模板侧：`amp_env_cfg.py` 224-240 —— `init` 的 params 有 `recovery_dir`；`reset` 的 params **根本没有** `recovery_dir` 键。

### 2. startup：读盘，并用 motion_dir 当 key 存两池

入口：`mdp/events.py` 193-206 `init_motion_loader()` → `MotionResetManager.init()`

关键存法（events.py 56、61 行）：

```
walk_run_frames[motion_dir] = 走路帧池      # key = WalkandRun 绝对路径
recovery_frames[motion_dir] = 倒地帧池      # key 也是 WalkandRun 路径，不是 Recovery 路径！
```

内存示意：

```
walk_run_frames[".../amp/WalkandRun"] = {root_pos: 7417帧, ...}
recovery_frames[".../amp/WalkandRun"] = {root_pos: 2575帧, ...}
```

`recovery_dir` 只在 events.py 53 行传给 `MotionLoader(..., recovery_dir=recovery_dir)` 用来读盘，读完即丢。

### 3. 每次 reset：只带 motion_dir 这把钥匙去查

入口：`mdp/events.py` 225-237 `reset_from_motion_data()` → `MotionResetManager.reset(motion_dir=...)`

分流（events.py 92-100 行）：

```
94  普通 env → walk_run_frames[motion_dir]
98  delay env → recovery_frames.get(motion_dir)   # 同一个 key 取出倒地池
```

### 4. 结论

reset 不传 `recovery_dir`，是因为：

1. `reset()` 函数签名（69-75 行）只有 `motion_dir` 参数；
2. 倒地张量已经不在磁盘路径上，而在 `recovery_frames[motion_dir]` 这个 dict 条目里；
3. 再传 `_recovery_dir` 字符串也查不到内存里的帧池。

自己验证：在 `events.py:61`（写入）和 `events.py:98`（读出）打 log，会看到同一个字符串 key（WalkandRun 绝对路径）。

---

## Q31: 把 recovery 改成独立用 recovery_dir 作 key（代码已改）

**改动：倒地帧池用 `recovery_dir` 作字典 key；reset 事件显式传 `recovery_dir`。**

### 改了哪些文件

1. `mdp/events.py`
   - init：`recovery_frames[recovery_dir] = ...`（原来误用 `motion_dir`）
   - reset：新增参数 `recovery_dir`，delay env 用 `recovery_frames.get(recovery_dir)`
   - `reset_from_motion_data`：新增 `recovery_dir` 并传给 reset

2. `amp_env_cfg.py`：`reset_from_motion.params` 增加 `"recovery_dir": None`

3. `config/g1/env_cfgs.py`：
```python
cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir
cfg.events["reset_from_motion"].params["recovery_dir"] = _recovery_dir
```

### 现在的内存结构

```
walk_run_frames[".../WalkandRun"] = 走路帧
recovery_frames[".../Recovery"]   = 倒地帧   # key 就是倒地目录
```

---

## Q32: 恢复辅助力课程（向上拽）

**规则：** delay 计数器到 50 时，80% 概率对 pelvis 施加世界系 +Z 力；力从 250N 起，每 500 训练轮减 20N，减到 0 后不再施力。play 也能看到绿色向上箭头。

### 实现

| 部分 | 位置 | 作用 |
|------|------|------|
| step 事件 | `mdp/events.py` → `apply_recovery_assist_force` | 读 `_delay_counters==50`，80% 触发，写 `xfrc_applied`，`debug_vis` 画箭头 |
| 课程 | `mdp/curriculums.py` → `recovery_assist_force` | `F = max(0, 250 - 20 * floor(iter/500))`，写回事件的 `current_force` |
| 配置 | `amp_env_cfg.py` + `g1/env_cfgs.py` | 事件挂 pelvis；课程 `steps_per_iter=24` |

### 时间线

```
delay_counter: 1..49 无辅助
             : 50    → 掷骰子，80% 开始向上 250N（随课程衰减）
             : 51..  持续施力直到 delay 结束 / reset
力衰减: iter 0→250N, 500→230N, …, 6500→0N
```

### play

- 保留 `recovery_assist_force` 事件，清空 curriculum → 力固定 250N
- `delay_reset_env_ratio=1.0`，倒地约 50 step 后可见绿色向上箭头（viewer 开 debug vis）

---

## Q33: AMP_mjlab 改为单 frame term（不再需要 history_ordering 补丁）

**改动：观测按 DroidUpE1 方式在函数内拼成一整帧，每 group 只留一个 term。**

### 现在的结构

| group | term | 函数 | history |
|-------|------|------|---------|
| actor | `frame` | `mdp.actor_frame` | 4 |
| critic | `frame` | `mdp.critic_frame` | 4 |
| amp | `state` | `mdp.amp_state` | 1 |

字段级噪声写在 `actor_frame` 里（与原先 UnoNoise 区间一致）；flat 地形用 `include_height_scan=False`。

### 为何不再需要补丁

单 term 时，mjlab 默认 `term` 展平 = 按时间排的帧历史，与原先 `history_ordering="time"` 多 term 交错结果一致。

---

## Q34: AMP_mjlab/rsl_rl 相对 TienKung-Lab-main/rsl_rl 改了什么

**核心算法（AMP 判别器、PPO 更新、lerp 奖励）没动，改动分四类：数据格式适配、数值稳定、ckpt 兼容、工程清理。**

### 1. AMPLoader 几乎重写（最大差异）

| | TienKung | AMP_mjlab |
|---|---|---|
| 数据格式 | JSON `Frames`（Dog 数据集，预拼接 20 关节+末端 12 维） | **.npz**（body_pos_w/body_quat_w/…/joint_pos，G1 30 body） |
| obs 构造 | 直接用文件里的扁平向量 | **逐帧算 anchor 局部系** pos_b/ori_b(mat[:,:2])/lin_vel_b/ang_vel_b，与 env 侧 `robot_body_*` obs 完全同构 |
| 采样 | 轨迹权重 + 时间插值(slerp) + 可预载 transitions | 按 motion 文件顺序循环 + 帧随机（无加权、无插值） |
| obs 维度 | 常量切片 (JOINT_POSE…END_POS) | `(3+6+3+3)*num_bodies` 属性 |

关键点：TienKung 的专家 obs 与策略侧 obs 无对齐约束；AMP_mjlab 把专家特征做成与仿真 obs **同一公式**，保证判别器两边可比。

### 2. amp_ppo.py（数值稳定）

- NaN/Inf 防护：returns/value/value_loss/loss 任一非有限 → **skip 该 mini-batch**（计数 `skipped_non_finite_batches`，log 可见）
- `min_std` 下限：每次 optimizer.step() 后把 policy std clamp 到 `min_normalized_std`（按 num_actions 自动补齐/截断）
- `effective_updates` 分母：用实际未跳过的更新数求均值
- `share_cnn_encoders`、`optimizer` 参数（预留）

### 3. actor_critic.py

- `std = clamp_min(std, 1e-6)`（scalar/log 两种）防 std→0 崩 Normal

### 4. amp_on_policy_runner.py（环境接口适配，核心是 mjlab 接入）

- `_migrate_train_cfg`：mjlab v5 actor/critic cfg → 旧版 policy cfg
- `_unpack_obs/_unpack_step`：TensorDict obs → legacy `(obs, extras)`，amp obs 从 extras 取
- terminal AMP state：mjlab 自动 reset → 用 pre-step amp_obs 近似终止态（TienKung 用 `env.get_amp_obs_for_expert_trans()`）
- ckpt 双格式加载：native `model_state_dict` 或 mjlab v5 `actor_state_dict(mlp.*)+critic`（键名映射 mlp.→actor./critic.，`distribution.std_param`→std）
- AMPLoader 调用改为 npz 参数集（body_names/anchor_name/all_body_names 从 env 解析）
- 导出 policy 函数兼容 TensorDict 输入

### 5. 杂项

- `discriminator.py`：纯格式化 + `reward.squeeze(-1)` 形状修正
- `rollout_storage`/`vec_env`/`on_policy_runner`：注释、空行级别
- 删了 `motion_loader_for_display.py`；打包改 pyproject

### 结论

AMP_mjlab = TienKung AMP 算法骨架 + **npz/mjlab 数据通路重写 + 训练稳定性加固 + v5 ckpt 兼容**。算法本身（判别器结构、AMP reward `clamp(1-(d-1)²/4)`、grad penalty、task_reward_lerp）一字未改。

---

## Q35: sim2sim 手柄档位控制脚本

**文件：`AMP_mjlab/scripts/sim2sim_gamepad.py`，自动找最新 log 里最新的 ONNX。**

- 观测：actor frame 96 维（imu_gyro + projected_gravity + command + joint_pos_rel + joint_vel_rel + last_action），history 4 → 384 维
- PD 参数 / default / action_scale / 关节顺序全部从 ONNX metadata 读取
- 力矩：`qfrc_applied` 直写（xml 无 actuator），±88N·m 限幅
- 手柄：左摇杆 X/Y → 侧移/前进档，右摇杆 X → 偏航档；**档位步进 0.5**（前进最多 6 档=3.0，后退 3 档=1.5，侧移 2 档=1.0，偏航 3 档=1.57）
- B=重置，A+B=退出；`--no-gamepad` 用固定 `--command`
- 运行：`python scripts/sim2sim_gamepad.py`（scene_g1.xml，含地面）
