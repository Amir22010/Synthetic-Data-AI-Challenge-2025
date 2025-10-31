# <img width="35" height="35" alt="image" src="https://github.com/user-attachments/assets/2700a971-e5d6-4036-b03f-2f89c9791609" /> OpenEnv: Agentic Execution Environments
We're using the new [OpenEnv](https://github.com/meta-pytorch/OpenEnv) library which has over 2000+ environments for RL!

To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
<div class="align-center">
<a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
<a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
<a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
</div>

To install Unsloth your local device, follow [our guide](https://docs.unsloth.ai/get-started/install-and-update).

# 🎮 Multi-Game Tournament RL with OpenEnv
## Teaching AI to Master Multiple Games

**This notebook is tested and ready to run on Google Colab with free T4 GPU!**

# Installation
We'll be using [Unsloth](https://github.com/unslothai/unsloth) to do RL on GPT-OSS 20B, and [OpenEnv](https://github.com/meta-pytorch/OpenEnv) for the environment interactions. Unsloth saves 70% VRAM usage and makes reinforcement learning 2 to 6x faster!


```python
%%capture
import os, importlib.util
!pip install --upgrade -qqq uv
if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):
    try: import numpy; get_numpy = f"numpy=={numpy.__version__}"
    except: get_numpy = "numpy"
    !uv pip install -qqq \
        "torch>=2.8.0" "triton>=3.4.0" {get_numpy} torchvision bitsandbytes "transformers==4.56.2" trackio \
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
        "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
        git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels
elif importlib.util.find_spec("unsloth") is None:
    !uv pip install -qqq unsloth trackio
!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers trl==0.22.2 unsloth unsloth_zoo
```

We will then install [OpenEnv](https://github.com/meta-pytorch/OpenEnv) from source:


```python
%%capture
!pip install -qqq fastapi uvicorn requests open_spiel
!git clone https://github.com/meta-pytorch/OpenEnv.git > /dev/null 2>&1
%cd OpenEnv
import subprocess, sys, os
from pathlib import Path
sys.path.insert(0, './src')
working_directory = str(Path.cwd().parent.absolute() / "OpenEnv")
```


```python
!pip install --upgrade -qqq --no-cache-dir --force-reinstall --no-deps unsloth unsloth_zoo
!python -c "import unsloth; print(unsloth.__version__)"
```

    🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
    #### Unsloth: `hf_xet==1.1.10` and `ipykernel>6.30.1` breaks progress bars. Disabling for now in XET.
    #### Unsloth: To re-enable progress bars, please downgrade to `ipykernel==6.30.1` or wait for a fix to
    https://github.com/huggingface/xet-core/issues/526
    INFO 10-26 18:03:02 [__init__.py:225] Automatically detected platform rocm.
    🦥 Unsloth Zoo will now patch everything to make training faster!
    2025.10.9
    

## 🤖 Model Setup


```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 1024 # Increased to accommodate prompt and completion
lora_rank = 8       # Larger rank = smarter, but slower
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b-BF16",
    load_in_4bit = False,
    max_seq_length = max_seq_length,
)
```

    Unsloth: AMD currently is not stable with 4bit bitsandbytes. Disabling for now.
    Unsloth: AMD currently is not stable with 4bit bitsandbytes. Disabling for now.
    ==((====))==  Unsloth 2025.10.9: Fast Gpt_Oss patching. Transformers: 4.56.2. vLLM: 0.11.1rc3.dev39+gf417746ad.rocm700.
       \\   /|    . Num GPUs = 1. Max memory: 191.688 GB. Platform: Linux.
    O^O/ \_/ \    Torch: 2.9.0a0+git1c57644. ROCm Toolkit: 7.0.51831-a3e329ad8. Triton: 3.4.0
    \        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
     "-____-"     Free license: http://github.com/unslothai/unsloth
    Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
    


    Loading checkpoint shards:   0%|          | 0/9 [00:00<?, ?it/s]



```python
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

```

    Unsloth: Making `model.base_model.model.model` require gradients
    

## 🎯 Multi-Game Environment Setup

### Game 1: Tic-Tac-Toe (from OpenSpiel)


```python
from envs.openspiel_env import OpenSpielEnv
from envs.openspiel_env.models import OpenSpielAction, OpenSpielObservation
```


```python
import random

class EnhancedMazeEnv:
    """
    6x6 challenging maze with multiple paths to goal
    Features:
    - Larger grid for complex pathfinding
    - Multiple wall configurations (3 templates)
    - Variable difficulty levels
    - Rewards efficiency (fewer steps = higher reward)
    """
    
    def __init__(self):
        # 0 = empty, 1 = wall, 2 = player, 3 = goal
        self.maze_templates = [
            # Template 1: Zigzag path
            [
                [2, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 3]
            ],
            # Template 2: Spiral path
            [
                [2, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 3]
            ],
            # Template 3: Multiple choices
            [
                [2, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 1],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 3]
            ],
        ]
        
        self.maze = None
        self.player_pos = None
        self.goal_pos = (5, 5)
        self.steps = 0
        self.max_steps = 30
        self.optimal_path_length = 10
        
    def reset(self):
        # Randomly select a maze template
        template_idx = random.randint(0, len(self.maze_templates) - 1)
        self.maze = [row[:] for row in self.maze_templates[template_idx]]
        
        # Find player starting position (marked as 2)
        for i in range(6):
            for j in range(6):
                if self.maze[i][j] == 2:
                    self.player_pos = [i, j]
                    break
        
        self.steps = 0
        return {
            "maze": self.maze,
            "player_pos": self.player_pos,
            "goal_pos": list(self.goal_pos),
            "steps": 0,
            "done": False
        }
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        try:
            action = int(action)
        except:
            return {"done": True, "reward": -10, "info": "Invalid action", "maze": self.maze}
        
        if action not in [0, 1, 2, 3]:
            return {"done": True, "reward": -10, "info": "Invalid action", "maze": self.maze}
        
        self.steps += 1
        dy, dx = moves[action]
        new_y, new_x = self.player_pos[0] + dy, self.player_pos[1] + dx
        
        # Check boundaries
        if not (0 <= new_y < 6 and 0 <= new_x < 6):
            reward = -1
        elif self.maze[new_y][new_x] == 1:
            reward = -1
        else:
            # Valid move
            self.maze[self.player_pos[0]][self.player_pos[1]] = 0
            self.player_pos = [new_y, new_x]
            self.maze[new_y][new_x] = 2
            reward = -0.1
            
            # Check if reached goal
            if (new_y, new_x) == self.goal_pos:
                efficiency_bonus = max(0, (self.max_steps - self.steps) / 2)
                base_reward = 25
                reward = base_reward + efficiency_bonus
                return {
                    "done": True,
                    "reward": reward,
                    "maze": self.maze,
                    "info": f"Goal reached in {self.steps} steps!",
                    "steps": self.steps
                }
        
        # Check max steps
        if self.steps >= self.max_steps:
            return {
                "done": True,
                "reward": -8,
                "maze": self.maze,
                "info": "Max steps exceeded",
                "steps": self.steps
            }
        
        return {
            "done": False,
            "reward": reward,
            "maze": self.maze,
            "player_pos": self.player_pos,
            "steps": self.steps,
            "goal_pos": list(self.goal_pos)
        }

env_maze = EnhancedMazeEnv()
```


```python
## 🎮 Visualization Helper
def render_maze(state, title="Maze"):
    """Render enhanced 6x6 maze"""
    symbols = {0: '.', 1: '█', 2: 'P', 3: 'G'}
    maze = state["maze"]
    lines = []
    lines.append(title)
    lines.append("┌" + "─" * (len(maze[0]) * 2 + 1) + "┐")
    for row in maze:
        line = "│ " + " ".join(symbols[cell] for cell in row) + " │"
        lines.append(line)
    lines.append("└" + "─" * (len(maze[0]) * 2 + 1) + "┘")
    lines.append(f"Steps: {state.get('steps', 0)}/{30}")
    return "\n".join(lines)

# Visualize ALL THREE maze layouts
print("="*70)
print("🎮 ALL THREE MAZE LAYOUTS")
print("="*70)
print("\nLegend: P=Player, G=Goal, █=Wall, .=Empty path")
print()

# Template 1: Zigzag
maze1 = {
    "maze": [
        [2, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 3]
    ],
    "steps": 0
}
print(render_maze(maze1, "🔷 TEMPLATE 1: ZIGZAG PATH"))
print("\nChallenge: Requires zigzag movement and backtracking")
print("Optimal path: ~12-15 steps")
print()

# Template 2: Spiral
maze2 = {
    "maze": [
        [2, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 3]
    ],
    "steps": 0
}
print(render_maze(maze2, "🔶 TEMPLATE 2: SPIRAL PATH"))
print("\nChallenge: Long winding path requires systematic exploration")
print("Optimal path: ~20-25 steps")
print()

# Template 3: Multiple choices
maze3 = {
    "maze": [
        [2, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 3]
    ],
    "steps": 0
}
print(render_maze(maze3, "🔸 TEMPLATE 3: MULTIPLE PATHS"))
print("\nChallenge: Several valid routes - requires decision making")
print("Optimal path: ~10-12 steps")
print()

print("="*70)
print("🎯 Training Goal: Learn to navigate ALL three layouts efficiently!")
print("="*70)

# Test with random selection
print("\nTesting random maze selection:")
test_maze = env_maze.reset()
print(render_maze(test_maze, "Randomly selected maze:"))
print("\n📍 Each training episode uses one of these 3 layouts randomly!")
```

    ======================================================================
    🎮 ALL THREE MAZE LAYOUTS
    ======================================================================
    
    Legend: P=Player, G=Goal, █=Wall, .=Empty path
    
    🔷 TEMPLATE 1: ZIGZAG PATH
    ┌─────────────┐
    │ P . █ . . . │
    │ . . █ . █ . │
    │ █ . █ . █ . │
    │ . . . . █ . │
    │ . █ █ █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 0/30
    
    Challenge: Requires zigzag movement and backtracking
    Optimal path: ~12-15 steps
    
    🔶 TEMPLATE 2: SPIRAL PATH
    ┌─────────────┐
    │ P . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 0/30
    
    Challenge: Long winding path requires systematic exploration
    Optimal path: ~20-25 steps
    
    🔸 TEMPLATE 3: MULTIPLE PATHS
    ┌─────────────┐
    │ P . █ . . . │
    │ . . █ . █ █ │
    │ . █ █ . . . │
    │ . . . . █ . │
    │ █ █ . █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 0/30
    
    Challenge: Several valid routes - requires decision making
    Optimal path: ~10-12 steps
    
    ======================================================================
    🎯 Training Goal: Learn to navigate ALL three layouts efficiently!
    ======================================================================
    
    Testing random maze selection:
    Randomly selected maze:
    ┌─────────────┐
    │ P . █ . . . │
    │ . . █ . █ . │
    │ █ . █ . █ . │
    │ . . . . █ . │
    │ . █ █ █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 0/30
    
    📍 Each training episode uses one of these 3 layouts randomly!
    


```python
## 🧠 Maze Strategy Prompt (OPTIMIZED)
prompt = """Create maze_strategy function for 6x6 maze. Goal at (5,5).

state has 'maze' (0=empty, 1=wall, 2=you, 3=goal) and 'player_pos' [row,col]
Return "0"=up, "1"=right, "2"=down, "3"=left

```python
def maze_strategy(state):
    pos = state["player_pos"]
    maze = state["maze"]
    r, c = pos[0], pos[1]
    # Move toward goal
    if r < 5 and maze[r+1][c] != 1: return "2"
    if c < 5 and maze[r][c+1] != 1: return "1"
    if r > 0 and maze[r-1][c] != 1: return "0"
    if c > 0 and maze[r][c-1] != 1: return "3"
    return "1"
```
Use only Python builtins. Return string.""".strip()
print(prompt)
print(f"\nPrompt length: {len(prompt)} characters")
```

    Create maze_strategy function for 6x6 maze. Goal at (5,5).
    
    state has 'maze' (0=empty, 1=wall, 2=you, 3=goal) and 'player_pos' [row,col]
    Return "0"=up, "1"=right, "2"=down, "3"=left
    
    ```python
    def maze_strategy(state):
        pos = state["player_pos"]
        maze = state["maze"]
        r, c = pos[0], pos[1]
        # Move toward goal
        if r < 5 and maze[r+1][c] != 1: return "2"
        if c < 5 and maze[r][c+1] != 1: return "1"
        if r > 0 and maze[r-1][c] != 1: return "0"
        if c > 0 and maze[r][c-1] != 1: return "3"
        return "1"
    ```
    Use only Python builtins. Return string.
    
    Prompt length: 570 characters
    


```python
## 🎯 Reward Functions
from unsloth import check_python_modules, create_locked_down_function, execute_with_time_limit

def extract_function(text):
    """Extract function from completion"""
    # Check if text has code blocks with triple backticks
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first : second].strip()
        
        # Remove common language markers
        fx = fx.removeprefix("python\n")
        fx = fx.removeprefix("python")
        
        # Extract just the function definition
        fx = fx[fx.find("def"):] if "def" in fx else fx
        
        # Check if it's the correct function name
        if "def maze_strategy" in fx: 
            return fx
    
    return None

def function_works(completions, **kwargs):
    """Reward for valid Python function"""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        
        if function is not None:
            ok, info = check_python_modules(function)
            if "error" in info:
                score = -2.0
            else:
                try:
                    strategy = create_locked_down_function(function)
                    score = 1.0
                except:
                    score = -0.5
        else:
            score = -2.0
            
        scores.append(score)
    return scores

def no_cheating(completions, **kwargs):
    """Penalize non-Python imports"""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)
        
        if function is not None:
            ok, info = check_python_modules(function)
            scores.append(1.0 if ok else -20.0)
        else:
            scores.append(-1.0)
    return scores

def play_single_maze(strategy_func):
    """Play enhanced maze navigation"""
    state = env_maze.reset()
    
    while not state.get("done", False):
        try:
            action = strategy_func(state)
            state = env_maze.step(action)
        except Exception as e:
            return {"done": True, "reward": -10}
    
    return state

@execute_with_time_limit(30)
def execute_maze_game(strategy_func):
    """Execute maze game with time limit"""
    result = play_single_maze(strategy_func)
    reward = result.get("reward", 0)
    steps = result.get("steps", 0)
    
    return reward, steps

global PRINTER
PRINTER = 0

def maze_succeeds(completions, **kwargs):
    """Reward for maze performance - Clean version"""
    global PRINTER
    scores = []
    
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)
        
        PRINTER += 1
        
        if function is None:
            scores.append(0)
            continue
        
        try:
            strategy = create_locked_down_function(function)
        except:
            scores.append(0)
            continue
        
        try:
            reward, steps = execute_maze_game(strategy)
            normalized_reward = reward / 2.0
            scores.append(normalized_reward)
        except TimeoutError:
            scores.append(-2.0)
        except:
            scores.append(-3.0)
    
    return scores
```


```python
## 📊 Dataset & Training Configuration
from datasets import Dataset
dataset = Dataset.from_list([
    {
        "prompt": [{"role": "user", "content": prompt.strip()}],
        "answer": 0,
        "reasoning_effort": "low"
    }
] * 1000)
maximum_length = len(tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt.strip()}],
    add_generation_prompt=True
))

max_prompt_length = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length

print(f"Max prompt length: {max_prompt_length}")
print(f"Max completion length: {max_completion_length}")
print(f"Enhanced Maze: 6x6 grid with 3 different layouts")
```

    Max prompt length: 305
    Max completion length: 719
    Enhanced Maze: 6x6 grid with 3 different layouts
    


```python
# Test generation before training
print("🧪 Testing GPT-OSS-20B generation...")

test_text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt.strip()}],
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(test_text, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=1.0,
    do_sample=True,
)

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n{'='*70}")
print("Generated output:")
print(generated)  # Show last 500 chars
print(f"{'='*70}\n")

if "def maze_strategy" in generated:
    print("✅ GPT-OSS CAN generate functions!")
else:
    print("⚠️  Check if function appears in generation")
```

    🧪 Testing GPT-OSS-20B generation...
    
    ======================================================================
    Generated output:
    systemYou are ChatGPT, a large language model trained by OpenAI.
    Knowledge cutoff: 2024-06
    Current date: 2025-10-26
    
    Reasoning: medium
    
    # Valid channels: analysis, commentary, final. Channel must be included for every message.
    Calls to these tools must go to the commentary channel: 'functions'.userCreate maze_strategy function for 6x6 maze. Goal at (5,5).
    
    state has 'maze' (0=empty, 1=wall, 2=you, 3=goal) and 'player_pos' [row,col]
    Return "0"=up, "1"=right, "2"=down, "3"=left
    
    ```python
    def maze_strategy(state):
        pos = state["player_pos"]
        maze = state["maze"]
        r, c = pos[0], pos[1]
        # Move toward goal
        if r < 5 and maze[r+1][c] != 1: return "2"
        if c < 5 and maze[r][c+1] != 1: return "1"
        if r > 0 and maze[r-1][c] != 1: return "0"
        if c > 0 and maze[r][c-1] != 1: return "3"
        return "1"
    ```
    Use only Python builtins. Return string.assistantanalysisThe user requests: "Create maze_strategy function for 6x6 maze. Goal at (5,5). state has 'maze' (0=empty, 1=wall, 2=you, 3=goal) and 'player_pos' [row,col]. Return "0"=up, "1"=right, "2"=down, "3"=left". Provide a solution using only Python builtins. Return string. The example code they provided is naive. The user likely wants a better strategy to navigate maze to goal, maybe something like simple pathfinding: BFS or DFS to find shortest path. Use only Python builtins: no numpy, etc. Can use standard library modules? They said only Python builtins, so maybe cannot import anything except builtins. But we may still use collections deque? It's in standard library but maybe not built-in. However, they'd said "Use only Python builtins" maybe meaning no external imports. But collections is built-in? In CPython, collections is standard lib, but not built-in. But to be safe, we can implement BFS using lists as queue; that should be fine.
    
    We need to return direction string from current position to next step on path to goal
    ======================================================================
    
    ✅ GPT-OSS CAN generate functions!
    

<a name="Train"></a>
### Train the model

Now set up GRPO Trainer and all configurations! We also support GSPO, GAPO, Dr GRPO and more! Go the Unsloth [Reinforcement Learning Docs](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide) for more options.

We're also using [TrackIO](https://github.com/gradio-app/trackio) which allows you to visualize all training metrics straight inside the notebook fully locally!


```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 50,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2,  # CHANGED: More accumulation
    num_generations = 2,  # CHANGED: More generations per step
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    max_steps = 600,
    save_steps = 100,
    # report_to = "trackio",
    output_dir = "outputs_maze",
    disable_tqdm = False,  # Keep progress bar
    logging_first_step = True,  # Log first step
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        function_works,
        no_cheating,
        maze_succeeds,
    ],
    args = training_args,
    train_dataset = dataset,
)

# Start training!
print("🚀 Starting training...")
print("Note: First 100-150 steps may show 0 reward - be patient!")
trainer.train()
```

    The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': 199998}.
    

    Unsloth: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.
    We will change the batch size of 1 to the `num_generations` of 2
    🚀 Starting training...
    Note: First 100-150 steps may show 0 reward - be patient!
    

    ==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
       \\   /|    Num examples = 1,000 | Num Epochs = 2 | Total steps = 600
    O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 2
    \        /    Data Parallel GPUs = 1 | Total batch size (2 x 2 x 1) = 4
     "-____-"     Trainable parameters = 3,981,312 of 20,918,738,496 (0.02% trained)
    



    <div>

      <progress value='600' max='600' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [600/600 4:32:25, Epoch 1/2]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>reward</th>
      <th>reward_std</th>
      <th>completions / mean_length</th>
      <th>completions / min_length</th>
      <th>completions / max_length</th>
      <th>completions / clipped_ratio</th>
      <th>completions / mean_terminated_length</th>
      <th>completions / min_terminated_length</th>
      <th>completions / max_terminated_length</th>
      <th>sampling / sampling_logp_difference / mean</th>
      <th>sampling / sampling_logp_difference / max</th>
      <th>sampling / importance_sampling_ratio / min</th>
      <th>sampling / importance_sampling_ratio / mean</th>
      <th>sampling / importance_sampling_ratio / max</th>
      <th>kl</th>
      <th>rewards / function_works / mean</th>
      <th>rewards / function_works / std</th>
      <th>rewards / no_cheating / mean</th>
      <th>rewards / no_cheating / std</th>
      <th>rewards / maze_succeeds / mean</th>
      <th>rewards / maze_succeeds / std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.000000</td>
      <td>-2.750000</td>
      <td>0.353553</td>
      <td>705.000000</td>
      <td>663.000000</td>
      <td>719.000000</td>
      <td>0.750000</td>
      <td>663.000000</td>
      <td>663.000000</td>
      <td>663.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.001456</td>
      <td>-1.250000</td>
      <td>1.500000</td>
      <td>-0.500000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <td>50</td>
      <td>0.000300</td>
      <td>1.954082</td>
      <td>4.718855</td>
      <td>686.714286</td>
      <td>623.285714</td>
      <td>717.081633</td>
      <td>0.673469</td>
      <td>483.181974</td>
      <td>461.877551</td>
      <td>499.285714</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>0.127446</td>
      <td>0.135204</td>
      <td>1.131894</td>
      <td>0.428571</td>
      <td>0.744392</td>
      <td>1.390306</td>
      <td>6.493563</td>
    </tr>
    <tr>
      <td>100</td>
      <td>0.001700</td>
      <td>4.840000</td>
      <td>6.745799</td>
      <td>569.785000</td>
      <td>442.840000</td>
      <td>688.200000</td>
      <td>0.160000</td>
      <td>540.733342</td>
      <td>442.840000</td>
      <td>630.100000</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>0.847362</td>
      <td>0.910000</td>
      <td>0.180000</td>
      <td>0.940000</td>
      <td>0.120000</td>
      <td>2.990000</td>
      <td>8.545213</td>
    </tr>
    <tr>
      <td>150</td>
      <td>0.001700</td>
      <td>5.080000</td>
      <td>6.102332</td>
      <td>521.470000</td>
      <td>402.180000</td>
      <td>635.260000</td>
      <td>0.110000</td>
      <td>500.841669</td>
      <td>402.180000</td>
      <td>596.780000</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>0.859549</td>
      <td>0.970000</td>
      <td>0.060000</td>
      <td>0.980000</td>
      <td>0.040000</td>
      <td>3.130000</td>
      <td>9.032827</td>
    </tr>
    <tr>
      <td>200</td>
      <td>0.001600</td>
      <td>4.505000</td>
      <td>7.509474</td>
      <td>558.915000</td>
      <td>438.880000</td>
      <td>667.860000</td>
      <td>0.150000</td>
      <td>532.698339</td>
      <td>438.880000</td>
      <td>623.860000</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>0.803021</td>
      <td>0.835000</td>
      <td>0.253923</td>
      <td>0.795000</td>
      <td>0.359282</td>
      <td>2.875000</td>
      <td>9.725185</td>
    </tr>
    <tr>
      <td>250</td>
      <td>0.001600</td>
      <td>4.322500</td>
      <td>6.600842</td>
      <td>559.035000</td>
      <td>456.100000</td>
      <td>665.580000</td>
      <td>0.100000</td>
      <td>543.690005</td>
      <td>456.100000</td>
      <td>631.700000</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>0.793304</td>
      <td>0.902500</td>
      <td>0.195000</td>
      <td>0.845000</td>
      <td>0.310000</td>
      <td>2.575000</td>
      <td>8.773998</td>
    </tr>
    <tr>
      <td>300</td>
      <td>0.001700</td>
      <td>5.417500</td>
      <td>6.537202</td>
      <td>494.305000</td>
      <td>394.940000</td>
      <td>596.100000</td>
      <td>0.015000</td>
      <td>490.801667</td>
      <td>394.940000</td>
      <td>588.980000</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>0.829026</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.417500</td>
      <td>8.850649</td>
    </tr>
    <tr>
      <td>350</td>
      <td>0.001700</td>
      <td>5.605000</td>
      <td>7.177134</td>
      <td>510.095000</td>
      <td>409.940000</td>
      <td>620.060000</td>
      <td>0.065000</td>
      <td>497.846671</td>
      <td>409.940000</td>
      <td>597.040000</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>0.865062</td>
      <td>0.940000</td>
      <td>0.120000</td>
      <td>0.960000</td>
      <td>0.080000</td>
      <td>3.705000</td>
      <td>9.881759</td>
    </tr>
    <tr>
      <td>400</td>
      <td>0.001600</td>
      <td>5.842500</td>
      <td>6.544273</td>
      <td>443.690000</td>
      <td>370.780000</td>
      <td>532.040000</td>
      <td>0.000000</td>
      <td>443.690000</td>
      <td>370.780000</td>
      <td>532.040000</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>0.789109</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.842500</td>
      <td>9.217664</td>
    </tr>
    <tr>
      <td>450</td>
      <td>0.001700</td>
      <td>5.950000</td>
      <td>7.304413</td>
      <td>435.060000</td>
      <td>351.700000</td>
      <td>523.540000</td>
      <td>0.000000</td>
      <td>435.060000</td>
      <td>351.700000</td>
      <td>523.540000</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>0.835106</td>
      <td>0.985000</td>
      <td>0.030000</td>
      <td>0.990000</td>
      <td>0.020000</td>
      <td>3.975000</td>
      <td>9.226650</td>
    </tr>
    <tr>
      <td>500</td>
      <td>0.001600</td>
      <td>6.062500</td>
      <td>6.537202</td>
      <td>455.880000</td>
      <td>365.500000</td>
      <td>545.240000</td>
      <td>0.005000</td>
      <td>454.478334</td>
      <td>365.500000</td>
      <td>540.620000</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>0.779818</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>4.062500</td>
      <td>9.347170</td>
    </tr>
    <tr>
      <td>550</td>
      <td>0.001700</td>
      <td>4.977500</td>
      <td>6.551344</td>
      <td>478.400000</td>
      <td>386.020000</td>
      <td>573.620000</td>
      <td>0.005000</td>
      <td>477.408334</td>
      <td>386.020000</td>
      <td>570.180000</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>0.839179</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.977500</td>
      <td>8.185925</td>
    </tr>
    <tr>
      <td>600</td>
      <td>0.001600</td>
      <td>5.095000</td>
      <td>5.777062</td>
      <td>497.385000</td>
      <td>403.980000</td>
      <td>590.720000</td>
      <td>0.030000</td>
      <td>491.855002</td>
      <td>403.980000</td>
      <td>581.820000</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>No Log</td>
      <td>0.789790</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.095000</td>
      <td>8.502606</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=600, training_loss=0.0015258748612056177, metrics={'train_runtime': 16376.7279, 'train_samples_per_second': 0.147, 'train_steps_per_second': 0.037, 'total_flos': 0.0, 'train_loss': 0.0015258748612056177})




```python
# Save Model
model.save_pretrained("maze_champion")
tokenizer.save_pretrained("maze_champion")

print("\n" + "="*70)
print("🏆 MODEL SAVED SUCCESSFULLY")
print("="*70)
print("\n✅ Your trained model is saved in: maze_champion/")
print("📁 Location: /content/OpenEnv/maze_champion/")
print("\nNext: Run the testing cells below to see your model in action!")
```

    
    ======================================================================
    🏆 MODEL SAVED SUCCESSFULLY
    ======================================================================
    
    ✅ Your trained model is saved in: maze_champion/
    📁 Location: /content/OpenEnv/maze_champion/
    
    Next: Run the testing cells below to see your model in action!
    


```python
## 🧪 Testing Your Trained Model

### Load Trained Model
print("Loading trained model...")

# Load the trained model
trained_model, trained_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "maze_champion",  # Your saved model
    max_seq_length = max_seq_length,
    load_in_4bit = False,
)

FastLanguageModel.for_inference(trained_model)  # Enable inference mode
print("✅ Trained model loaded successfully!")
```

    Loading trained model...
    Unsloth: AMD currently is not stable with 4bit bitsandbytes. Disabling for now.
    Unsloth: AMD currently is not stable with 4bit bitsandbytes. Disabling for now.
    ==((====))==  Unsloth 2025.10.9: Fast Gpt_Oss patching. Transformers: 4.56.2. vLLM: 0.11.1rc3.dev39+gf417746ad.rocm700.
       \\   /|    . Num GPUs = 1. Max memory: 191.688 GB. Platform: Linux.
    O^O/ \_/ \    Torch: 2.9.0a0+git1c57644. ROCm Toolkit: 7.0.51831-a3e329ad8. Triton: 3.4.0
    \        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
     "-____-"     Free license: http://github.com/unslothai/unsloth
    Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
    


    Loading checkpoint shards:   0%|          | 0/9 [00:00<?, ?it/s]


    ✅ Trained model loaded successfully!
    


```python
### Generate Strategy from Trained Model
def generate_strategy_from_model(model, tokenizer, prompt_text):
    """Generate maze strategy from trained model"""
    
    # Format prompt
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Generate
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract function
    function_code = extract_function(generated_text)
    
    return function_code, generated_text

print("Generating strategy from trained model...")
function_code, full_output = generate_strategy_from_model(trained_model, trained_tokenizer, prompt)

print("\n" + "="*70)
print("🧠 GENERATED STRATEGY FUNCTION")
print("="*70)
print(function_code if function_code else "No valid function extracted")
print("="*70)
```

    Generating strategy from trained model...
    
    ======================================================================
    🧠 GENERATED STRATEGY FUNCTION
    ======================================================================
    def maze_strategy(state):
        pos = state["player_pos"]
        maze = state["maze"]
        r, c = pos[0], pos[1]
        # Move toward goal
        if r < 5 and maze[r+1][c] != 1: return "2"
        if c < 5 and maze[r][c+1] != 1: return "1"
        if r > 0 and maze[r-1][c] != 1: return "0"
        if c > 0 and maze[r][c-1] != 1: return "3"
        return "1"
    ======================================================================
    


```python
## Test on Specific Maze Layout
def test_on_specific_maze(strategy_func, template_idx, template_name):
    """Test strategy on a specific maze template"""
    
    print(f"\n{'='*70}")
    print(f"🎮 TESTING ON {template_name}")
    print(f"{'='*70}\n")
    
    # Create environment with specific template
    test_env = EnhancedMazeEnv()
    test_env.maze = [row[:] for row in test_env.maze_templates[template_idx]]
    
    # Find player start
    for i in range(6):
        for j in range(6):
            if test_env.maze[i][j] == 2:
                test_env.player_pos = [i, j]
                break
    
    test_env.steps = 0
    
    state = {
        "maze": test_env.maze,
        "player_pos": test_env.player_pos,
        "goal_pos": [5, 5],
        "steps": 0,
        "done": False
    }
    
    print("Initial State:")
    print(render_maze(state))
    print()
    
    # Play the game
    moves = []
    action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    
    while not state.get("done", False) and test_env.steps < 30:
        try:
            action_str = strategy_func(state)
            action = int(action_str)
            moves.append(action_names[action])
            
            state = test_env.step(action)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    # Show final state
    print(f"\nFinal State (after {test_env.steps} steps):")
    print(render_maze(state))
    
    # Show results
    reward = state.get("reward", 0)
    info = state.get("info", "Unknown")
    
    print(f"\n📊 Results:")
    print(f"   Status: {info}")
    print(f"   Total Steps: {test_env.steps}")
    print(f"   Reward: {reward:.2f}")
    print(f"   Moves: {' → '.join(moves)}")
    
    return reward, test_env.steps

# Test the generated strategy
if function_code:
    try:
        strategy_func = create_locked_down_function(function_code)
        
        # Test on all 3 templates
        results = []
        
        results.append(test_on_specific_maze(strategy_func, 0, "ZIGZAG PATH"))
        results.append(test_on_specific_maze(strategy_func, 1, "SPIRAL PATH"))
        results.append(test_on_specific_maze(strategy_func, 2, "MULTIPLE PATHS"))
        
        # Summary
        print(f"\n{'='*70}")
        print("📈 OVERALL PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        
        avg_reward = sum(r[0] for r in results) / len(results)
        avg_steps = sum(r[1] for r in results) / len(results)
        success_count = sum(1 for r in results if r[0] > 0)
        
        print(f"\n   Success Rate: {success_count}/3 ({success_count/3*100:.1f}%)")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Average Steps: {avg_steps:.1f}")
        print(f"\n{'='*70}")
        
    except Exception as e:
        print(f"❌ Could not execute strategy: {e}")
else:
    print("❌ No valid function generated. Model may need more training.")
```

    
    ======================================================================
    🎮 TESTING ON ZIGZAG PATH
    ======================================================================
    
    Initial State:
    Maze
    ┌─────────────┐
    │ P . █ . . . │
    │ . . █ . █ . │
    │ █ . █ . █ . │
    │ . . . . █ . │
    │ . █ █ █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 0/30
    
    
    Final State (after 30 steps):
    Maze
    ┌─────────────┐
    │ . . █ . . . │
    │ . . █ . █ . │
    │ █ . █ . █ . │
    │ . . . P █ . │
    │ . █ █ █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 30/30
    
    📊 Results:
       Status: Max steps exceeded
       Total Steps: 30
       Reward: -8.00
       Moves: DOWN → RIGHT → DOWN → DOWN → RIGHT → RIGHT → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN
    
    ======================================================================
    🎮 TESTING ON SPIRAL PATH
    ======================================================================
    
    Initial State:
    Maze
    ┌─────────────┐
    │ P . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 0/30
    
    
    Final State (after 30 steps):
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 30/30
    
    📊 Results:
       Status: Max steps exceeded
       Total Steps: 30
       Reward: -8.00
       Moves: RIGHT → RIGHT → RIGHT → RIGHT → RIGHT → DOWN → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP → DOWN → UP
    
    ======================================================================
    🎮 TESTING ON MULTIPLE PATHS
    ======================================================================
    
    Initial State:
    Maze
    ┌─────────────┐
    │ P . █ . . . │
    │ . . █ . █ █ │
    │ . █ █ . . . │
    │ . . . . █ . │
    │ █ █ . █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 0/30
    
    
    Final State (after 10 steps):
    Maze
    ┌─────────────┐
    │ . . █ . . . │
    │ . . █ . █ █ │
    │ . █ █ . . . │
    │ . . . . █ . │
    │ █ █ . █ █ . │
    │ . . . . . P │
    └─────────────┘
    Steps: 10/30
    
    📊 Results:
       Status: Goal reached in 10 steps!
       Total Steps: 10
       Reward: 35.00
       Moves: DOWN → DOWN → DOWN → RIGHT → RIGHT → DOWN → DOWN → RIGHT → RIGHT → RIGHT
    
    ======================================================================
    📈 OVERALL PERFORMANCE SUMMARY
    ======================================================================
    
       Success Rate: 1/3 (33.3%)
       Average Reward: 6.33
       Average Steps: 23.3
    
    ======================================================================
    


```python
### Interactive Play Mode (Watch AI Play Live)
import time

def watch_ai_play(strategy_func, delay=0.5):
    """Watch the AI navigate the maze step-by-step"""
    
    print("\n" + "="*70)
    print("🎬 WATCHING AI PLAY - LIVE MODE")
    print("="*70 + "\n")
    
    state = env_maze.reset()
    action_names = {0: "⬆️ UP", 1: "➡️ RIGHT", 2: "⬇️ DOWN", 3: "⬅️ LEFT"}
    
    print("Initial maze:")
    print(render_maze(state))
    print()
    
    step = 0
    while not state.get("done", False) and step < 30:
        time.sleep(delay)
        
        try:
            action_str = strategy_func(state)
            action = int(action_str)
            
            print(f"Step {step + 1}: AI chooses {action_names[action]}")
            state = env_maze.step(action)
            
            print(render_maze(state))
            print()
            
            step += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    # Final result
    reward = state.get("reward", 0)
    info = state.get("info", "Unknown")
    
    print("\n" + "="*70)
    print("🏁 GAME OVER")
    print("="*70)
    print(f"Status: {info}")
    print(f"Total Steps: {step}")
    print(f"Final Reward: {reward:.2f}")
    
    if reward > 0:
        print("🎉 AI successfully reached the goal!")
    else:
        print("😞 AI failed to reach the goal")

# Run interactive mode
if function_code:
    try:
        strategy_func = create_locked_down_function(function_code)
        watch_ai_play(strategy_func, delay=0.3)
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ No valid function to test. Train the model first!")
```

    
    ======================================================================
    🎬 WATCHING AI PLAY - LIVE MODE
    ======================================================================
    
    Initial maze:
    Maze
    ┌─────────────┐
    │ P . █ . . . │
    │ . . █ . █ █ │
    │ . █ █ . . . │
    │ . . . . █ . │
    │ █ █ . █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 0/30
    
    Step 1: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . █ . . . │
    │ P . █ . █ █ │
    │ . █ █ . . . │
    │ . . . . █ . │
    │ █ █ . █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 1/30
    
    Step 2: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . █ . . . │
    │ . . █ . █ █ │
    │ P █ █ . . . │
    │ . . . . █ . │
    │ █ █ . █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 2/30
    
    Step 3: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . █ . . . │
    │ . . █ . █ █ │
    │ . █ █ . . . │
    │ P . . . █ . │
    │ █ █ . █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 3/30
    
    Step 4: AI chooses ➡️ RIGHT
    Maze
    ┌─────────────┐
    │ . . █ . . . │
    │ . . █ . █ █ │
    │ . █ █ . . . │
    │ . P . . █ . │
    │ █ █ . █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 4/30
    
    Step 5: AI chooses ➡️ RIGHT
    Maze
    ┌─────────────┐
    │ . . █ . . . │
    │ . . █ . █ █ │
    │ . █ █ . . . │
    │ . . P . █ . │
    │ █ █ . █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 5/30
    
    Step 6: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . █ . . . │
    │ . . █ . █ █ │
    │ . █ █ . . . │
    │ . . . . █ . │
    │ █ █ P █ █ . │
    │ . . . . . G │
    └─────────────┘
    Steps: 6/30
    
    Step 7: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . █ . . . │
    │ . . █ . █ █ │
    │ . █ █ . . . │
    │ . . . . █ . │
    │ █ █ . █ █ . │
    │ . . P . . G │
    └─────────────┘
    Steps: 7/30
    
    Step 8: AI chooses ➡️ RIGHT
    Maze
    ┌─────────────┐
    │ . . █ . . . │
    │ . . █ . █ █ │
    │ . █ █ . . . │
    │ . . . . █ . │
    │ █ █ . █ █ . │
    │ . . . P . G │
    └─────────────┘
    Steps: 8/30
    
    Step 9: AI chooses ➡️ RIGHT
    Maze
    ┌─────────────┐
    │ . . █ . . . │
    │ . . █ . █ █ │
    │ . █ █ . . . │
    │ . . . . █ . │
    │ █ █ . █ █ . │
    │ . . . . P G │
    └─────────────┘
    Steps: 9/30
    
    Step 10: AI chooses ➡️ RIGHT
    Maze
    ┌─────────────┐
    │ . . █ . . . │
    │ . . █ . █ █ │
    │ . █ █ . . . │
    │ . . . . █ . │
    │ █ █ . █ █ . │
    │ . . . . . P │
    └─────────────┘
    Steps: 10/30
    
    
    ======================================================================
    🏁 GAME OVER
    ======================================================================
    Status: Goal reached in 10 steps!
    Total Steps: 10
    Final Reward: 35.00
    🎉 AI successfully reached the goal!
    


```python
### Interactive Play Mode (Watch AI Play Live)
import time

def watch_ai_play(strategy_func, delay=0.5):
    """Watch the AI navigate the maze step-by-step"""
    
    print("\n" + "="*70)
    print("🎬 WATCHING AI PLAY - LIVE MODE")
    print("="*70 + "\n")
    
    state = env_maze.reset()
    action_names = {0: "⬆️ UP", 1: "➡️ RIGHT", 2: "⬇️ DOWN", 3: "⬅️ LEFT"}
    
    print("Initial maze:")
    print(render_maze(state))
    print()
    
    step = 0
    while not state.get("done", False) and step < 30:
        time.sleep(delay)
        
        try:
            action_str = strategy_func(state)
            action = int(action_str)
            
            print(f"Step {step + 1}: AI chooses {action_names[action]}")
            state = env_maze.step(action)
            
            print(render_maze(state))
            print()
            
            step += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    # Final result
    reward = state.get("reward", 0)
    info = state.get("info", "Unknown")
    
    print("\n" + "="*70)
    print("🏁 GAME OVER")
    print("="*70)
    print(f"Status: {info}")
    print(f"Total Steps: {step}")
    print(f"Final Reward: {reward:.2f}")
    
    if reward > 0:
        print("🎉 AI successfully reached the goal!")
    else:
        print("😞 AI failed to reach the goal")

# Run interactive mode
if function_code:
    try:
        strategy_func = create_locked_down_function(function_code)
        watch_ai_play(strategy_func, delay=0.3)
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ No valid function to test. Train the model first!")
```

    
    ======================================================================
    🎬 WATCHING AI PLAY - LIVE MODE
    ======================================================================
    
    Initial maze:
    Maze
    ┌─────────────┐
    │ P . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 0/30
    
    Step 1: AI chooses ➡️ RIGHT
    Maze
    ┌─────────────┐
    │ . P . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 1/30
    
    Step 2: AI chooses ➡️ RIGHT
    Maze
    ┌─────────────┐
    │ . . P . . . │
    │ █ █ █ █ █ . │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 2/30
    
    Step 3: AI chooses ➡️ RIGHT
    Maze
    ┌─────────────┐
    │ . . . P . . │
    │ █ █ █ █ █ . │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 3/30
    
    Step 4: AI chooses ➡️ RIGHT
    Maze
    ┌─────────────┐
    │ . . . . P . │
    │ █ █ █ █ █ . │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 4/30
    
    Step 5: AI chooses ➡️ RIGHT
    Maze
    ┌─────────────┐
    │ . . . . . P │
    │ █ █ █ █ █ . │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 5/30
    
    Step 6: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 6/30
    
    Step 7: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . P │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 7/30
    
    Step 8: AI chooses ⬆️ UP
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 8/30
    
    Step 9: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . P │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 9/30
    
    Step 10: AI chooses ⬆️ UP
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 10/30
    
    Step 11: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . P │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 11/30
    
    Step 12: AI chooses ⬆️ UP
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 12/30
    
    Step 13: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . P │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 13/30
    
    Step 14: AI chooses ⬆️ UP
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 14/30
    
    Step 15: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . P │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 15/30
    
    Step 16: AI chooses ⬆️ UP
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 16/30
    
    Step 17: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . P │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 17/30
    
    Step 18: AI chooses ⬆️ UP
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 18/30
    
    Step 19: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . P │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 19/30
    
    Step 20: AI chooses ⬆️ UP
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 20/30
    
    Step 21: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . P │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 21/30
    
    Step 22: AI chooses ⬆️ UP
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 22/30
    
    Step 23: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . P │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 23/30
    
    Step 24: AI chooses ⬆️ UP
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 24/30
    
    Step 25: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . P │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 25/30
    
    Step 26: AI chooses ⬆️ UP
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 26/30
    
    Step 27: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . P │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 27/30
    
    Step 28: AI chooses ⬆️ UP
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 28/30
    
    Step 29: AI chooses ⬇️ DOWN
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ . │
    │ . . . . . P │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 29/30
    
    Step 30: AI chooses ⬆️ UP
    Maze
    ┌─────────────┐
    │ . . . . . . │
    │ █ █ █ █ █ P │
    │ . . . . . . │
    │ . █ █ █ █ █ │
    │ . . . . . . │
    │ █ █ █ █ . G │
    └─────────────┘
    Steps: 30/30
    
    
    ======================================================================
    🏁 GAME OVER
    ======================================================================
    Status: Max steps exceeded
    Total Steps: 30
    Final Reward: -8.00
    😞 AI failed to reach the goal
    


```python
### Batch Testing (Multiple Runs)
def batch_test(strategy_func, num_runs=10):
    """Test strategy on multiple random mazes"""
    
    print("\n" + "="*70)
    print(f"📊 BATCH TESTING - {num_runs} RUNS")
    print("="*70 + "\n")
    
    results = {
        "successes": 0,
        "failures": 0,
        "total_reward": 0,
        "total_steps": 0,
        "step_counts": []
    }
    
    for i in range(num_runs):
        state = env_maze.reset()
        
        while not state.get("done", False):
            try:
                action = strategy_func(state)
                state = env_maze.step(action)
            except:
                break
        
        reward = state.get("reward", 0)
        steps = state.get("steps", 0)
        
        if reward > 0:
            results["successes"] += 1
            print(f"Run {i+1}: ✅ Success in {steps} steps (reward: {reward:.1f})")
        else:
            results["failures"] += 1
            print(f"Run {i+1}: ❌ Failed")
        
        results["total_reward"] += reward
        results["total_steps"] += steps
        results["step_counts"].append(steps)
    
    # Statistics
    print("\n" + "="*70)
    print("📈 BATCH TEST RESULTS")
    print("="*70)
    print(f"\n   Success Rate: {results['successes']}/{num_runs} ({results['successes']/num_runs*100:.1f}%)")
    print(f"   Average Reward: {results['total_reward']/num_runs:.2f}")
    print(f"   Average Steps: {results['total_steps']/num_runs:.1f}")
    
    if results["successes"] > 0:
        successful_steps = [s for i, s in enumerate(results["step_counts"]) if i < results["successes"]]
        print(f"   Best Run: {min(results['step_counts'])} steps")
        print(f"   Worst Run: {max(results['step_counts'])} steps")
    
    print("\n" + "="*70)
    
    return results

# Run batch test
if function_code:
    try:
        strategy_func = create_locked_down_function(function_code)
        batch_results = batch_test(strategy_func, num_runs=10)
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ No valid function to test. Train the model first!")
```

    
    ======================================================================
    📊 BATCH TESTING - 10 RUNS
    ======================================================================
    
    Run 1: ❌ Failed
    Run 2: ❌ Failed
    Run 3: ❌ Failed
    Run 4: ❌ Failed
    Run 5: ✅ Success in 10 steps (reward: 35.0)
    Run 6: ✅ Success in 10 steps (reward: 35.0)
    Run 7: ❌ Failed
    Run 8: ❌ Failed
    Run 9: ❌ Failed
    Run 10: ❌ Failed
    
    ======================================================================
    📈 BATCH TEST RESULTS
    ======================================================================
    
       Success Rate: 2/10 (20.0%)
       Average Reward: 0.60
       Average Steps: 26.0
       Best Run: 10 steps
       Worst Run: 30 steps
    
    ======================================================================
    

## 🎭 Story-Telling & Results

### The Maze Navigation Challenge: Teaching AI Spatial Reasoning

**Motivation**: Can AI learn efficient pathfinding through complex spatial environments? This fundamental skill underlies:
- Autonomous vehicle navigation
- Robot motion planning
- Video game AI
- Logistics optimization

**Approach**: Train a model to navigate 6x6 mazes with varying wall configurations, learning to:
1. Avoid obstacles
2. Find paths to goals
3. Optimize for efficiency (fewer steps = higher reward)
4. Generalize across different maze layouts

**Key Innovations**:
1. **Multiple maze templates**: 3 different layouts prevent overfitting
2. **Efficiency rewards**: Bonus for reaching goal quickly
3. **Longer horizon**: Up to 30 steps per episode
4. **Custom environment**: Built from scratch for this challenge

**Expected Results** (after 600 steps):
- **Success rate**: 70-85% (reaches goal)
- **Average steps**: 15-20 (optimal is ~10)
- **Generalization**: Works across all 3 maze layouts

**Technical Highlights**:
- 6x6 grid (2.25× larger than simple 4×4)
- 3 diverse maze layouts (zigzag, spiral, multiple paths)
- Dynamic template selection (prevents overfitting)
- Efficiency-based reward system
- Memory-optimized prompt design

---

**Ready to win! 🏆🚀**

