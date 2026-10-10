# 本地使用说明

这份说明只用于 `local/learning-poc` 分支。这份说明讲三件事：怎么启动游戏、怎么让 Jev 玩、怎么让本地模型玩。

## 1. 准备（只做一次）

这些东西已经装好了。换电脑时，按下面的步骤重新装。

1. 安装依赖：
   ```bash
   npm install
   ```
2. 安装并启动 Ollama（需要 0.35 以上）：
   ```bash
   brew install ollama
   brew services start ollama      # 开机自动运行；不想要时用 brew services stop ollama
   ```
3. 下载训练好的模型，并导入 Ollama：
   ```bash
   hf download grapeot/decision-pacman-0.8b-GGUF pacman-0.8b-qwen-Q8_0.gguf Modelfile.pacman-0.8b-qwen --local-dir ./tmp-model
   cd tmp-model
   echo 'CAPABILITY decision' >> Modelfile.pacman-0.8b-qwen
   ollama create pacman-0.8b-qwen -f Modelfile.pacman-0.8b-qwen
   cd .. && rm -rf tmp-model      # Ollama 已经复制了模型，下载的文件可以删掉
   ```
   `hf download` 卡住的话，可以用 `curl -L -C -` 下载同一个文件。`-C -` 让 curl 从断开的地方接着下。
4. 在项目根目录建一个 `.env` 文件。git 会忽略这个文件，所以 key 不会被提交：
   ```bash
   VITE_DECISION_BASE_URL=http://localhost:11434
   VITE_DEFAULT_MODEL=pacman-0.8b-qwen
   TYPESAFE_API_KEY=<你的 TypeSafe key>
   ```

## 2. 启动游戏

```bash
npm run dev
```

然后用浏览器打开 http://localhost:5173/ 。

开发服务器 (dev server) 会自己读 `.env` 里的 key。key 只在开发服务器里使用，不会进到网页里。

## 3. 在浏览器里选玩家

右边面板的 **Player** 选 `Model`。然后在 **Model** 输入框里选一个玩家：

| Model 里填的名字 | 谁在玩 | 请求发到哪里 |
|---|---|---|
| `pacman-0.8b-qwen` | 训练后的 0.8B 模型（默认） | `/decide` → 本地 Ollama |
| `jev-latest` | TypeSafe 托管的 Jev | `/jev` → `https://api.typesafe.ai` |
| `random` | 每次被问都随机选（会在原地抖） | 不发请求 |
| `random-1s` | 每个路口随机选出口；每 1 秒按 3:1 决定往前还是掉头 | 不发请求 |
| `random-2s` | 同上，每 2 秒决定一次 | 不发请求 |

**怎么打开下拉建议：** 先把 Model 框里的文字删掉，再点一下这个框。Chrome 只显示和框里文字匹配的建议。选好后按回车。

**Endpoint 框保持 `/decide`，不要改。** 页面选 `jev-latest` 时会自动走 `/jev`。

Player 选 `Keyboard` 时，你自己用方向键或 WASD 玩。空格暂停，R 重新开始，M 静音。

### 也可以直接用网址选

- 本地模型：http://localhost:5173/?model=pacman-0.8b-qwen
- Jev：http://localhost:5173/?model=jev-latest
- 随机走：http://localhost:5173/?model=random-1s
- 自己玩：http://localhost:5173/?mode=human

网址里还可以加这些参数：
- `&wait=1`：游戏在路口停下，等模型回答完再走。这样慢模型也能被公平地评价。
- `&speed=2`：游戏速度变成 2 倍。
- `&seed=100`：固定这局的随机种子，每次开局都一样。
- `&sound=0`：关掉声音。

## 4. 在命令行跑无界面对局 (headless)

命令行会自己读 `.env`。每局的记录写在 `runs/` 目录里。

```bash
# 本地模型
npm run headless -- --policy pacman-0.8b-qwen --games 1 --seed 100 --max-seconds 60

# Jev（命令行里的名字是 jev，不是 jev-latest）
npm run headless -- --policy jev --games 1 --seed 100 --max-seconds 60

# 随机走、手写规则
npm run headless -- --policy random-1s --games 1 --seed 100 --max-seconds 60
npm run headless -- --policy greedy --games 1 --seed 100 --max-seconds 60
```

### 和 README 排行榜一样的对比

README 的排行榜用种子 100-109，一共 10 局，每局最多 5 分钟：

```bash
npm run headless -- --policy random-1s --games 10 --seed 100 --max-seconds 300
```

命令最后一行打印 JSON 总结。`meanPellets` 是平均吃到的豆子数。

## 5. 在这台电脑上测到的数字

电脑：Apple M4，16 GB 内存。种子 100-109，10 局，每局最多 5 分钟。

| 玩家 | 平均豆子 |
|---|---|
| `random` | 27.6 |
| `random-1s` | 130.1 |
| `random-2s` | 144.6 |
| `greedy` | 200.7 |

本地模型 `pacman-0.8b-qwen` 在这台电脑上每次决策大约 260–370 ms。README 里的 55 ms 是在更快的电脑上测的。

## 6. 注意事项

- **Jev 会花钱。** Jev 是托管服务，游戏开着时每秒大约发 4–5 次请求。不用的时候把 Player 切到 `Keyboard`，或者关掉页面。
- **key 只放在 `.env` 里。** 不要把 key 写进任何会被提交的文件。
- **本地模型连不上时：** 运行 `curl localhost:11434/api/version` 看 Ollama 有没有在运行。没在运行的话，运行 `brew services start ollama`。
- **想看 `tev1:4b`：** 先运行 `ollama pull tev1:4b` 下载，这个模型有几 GB。然后在 Model 框里填 `tev1:4b`。
