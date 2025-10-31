<h1 align='center'>✨ Welcome to the Synthetic Data AI Agents Challenge ✨</h1>
<h2 align='center'>Hosted by AMD, Pytorch, and Unsloth</h2>


---
## Task
You will be building:
1.  **A question agent** that will ask $N$ puzzle-based questions based on provided [topics](./assets/topics.json).
    - Create your model in [question_model.py](./agents/question_model.py) (it will be called by [question_agent.py](./agents/question_agent.py) for evaluation)
    - *Your question agent must output questions in the format specified in [sample_question.json](./assets/sample_question.json)*.
2. **An answer agent** that answers questions asked from a question agent.
    -  Create your model in [answer_model.py](./agents/answer_model.py) (it will be called by [answer_agent.py](./agents/answer_agent.py) for evaluation)
    -  *Your answer agent must output answers in the format specified in [sample_answer.json](./assets/sample_answer.json)*.
---


## Instructions

1. Read through this README.ipynb for more details on the challenge.
    - **Note:** If members of your team are working from the notebook simultaneously, please coordinate to ensure you do not overwrite each other's work.
1. Check out our [Synthetic Data Generation and Unsloth Tutorial](./tutorial.ipynb) for training tips and tricks.

## 📚 Table of Contents:
- 📝 [Task](#task)
- ⚙️ [Instructions](#instructions)
- 🏏 [Tournament Overview](#tournament-overview)
- 📋 [Guidelines](#guidelines)
    - [Format](#format-overview)
- 🛠️ [Submission](#️what-you-will-submit)
- ⚠️ [Restrictions](#restrictions)
- 📂 [Directory & Files overview](#directory--files-overview)
- 🎮 [Getting started](#getting-started)
    - 🚀 [Env Setup](#env-setup)
    - 🤔 [Q-Agent](#q-agent)
        - ✅ [Basic format-checks for questions from Q-agent](#basic-format-checks-for-questions-from-q-agent)
    - 🤖 [A-agent](#a-agent)
        - ✅ [Basic format-checks for answers from A-agent](#basic-format-checks-for-answers-from-a-agent)
- 🏅 [Evaluation](#evaluation)
    - 📊 [Scoring Criteria](#scoring-criteria)
    - 🧮 [Scoring Example](#scoring-example)
- ⏱ [Time Limit](#time-limit)
<!-- - 🏆 [LeaderBoard UI/UX](#leaderboard-uiux) -->

## Tournament Overview
<!-- 🏏  -->
* All matches in this tournament will be **1v1** knockout format where two teams, Team-A vs Team-B, will compete with their Q-agent (question agent) and A-agent (answer agent). You can think of this as a cricket match or baseball game where teams will switch sides.
* Like in cricket, each match has two innings:
    -   1st inning:
        *   $N$ Question from the Q-agent (Team-A) and their corresponding $N$ answers from the A-agent (Team-B).
        *   Q-agent score (Team-A): Say, $40$
        *   A-agent score (Team-B): $60$

    -   2nd inning:
        *   $N$ Question from the Q-agent (Team-B) and their respective $N$ responses from the A-agent (Team-A).
        *   Q-agent score (Team-B): Say, $70$
        *   A-agent score (Team-A): $30$
    -   Final Score:
        *   Team-A score $=$ 1st inning Q-agent score $+$ 2nd inning A-agent score $= 40 + 30 = 70$
        *   Team-B score $=$ 1st inning A-agent score $+$ 2nd inning Q-agent score $= 60 + 70 = 130$

    -   Winner: **Team-B** with a score of $130$.

For more info on how scoring is done, refer to the [scoring criteria section](#scoring-criteria).


## Guidelines
<!-- 📋  -->

### Format
We will only consider responses from the Q-agent and the A-agent which follow the below format.

*Note*: While having an explanation/reasoning is a plus, not having them doesn't disqualify the question or answer being correct.

#### Q-Agent
Given a topic, the Q-agent should generate questions in the specified JSON format:

```json
{
    "topic": "<Topic of the Question>",
    "question": "<full question text>",
    "choices": [
        "A) <choice A text>",
        "B) <choice B text>",
        "C) <choice C text>",
        "D) <choice D text>"
    ],
    "answer": "<correct choice letter only>",
    "explanation": "brief explanation within 100 words for why the answer is correct"
}
```

The **"Topic"**, **"Question"**, **"Choices"**, and **"Answer"** will be verified for correctness.

#### A-Agent
Given a Question and Choices, A-agent should produce answer in the format of:

```json
{
    "answer": "<correct choice letter only>",
    "reasoning": "brief reasoning within 100 words for why the answer is correct"
}
```

The **"Answer"** key will be compared with **"Answer"** from the opponent's Q-agent.

## Submission
<!-- 🛠️  -->
You need to submit your code which should contain these main files:
1. All work must be within the `AIAC` folder. Do NOT change the folder name.
1. No need to upload anything anywhere, we'll collect your agent code from your Jupyter Server at the end of the challenge.
   1. The agents will be called by `python -m agents.question_agent` and `python -m agents.answer_agent`, respectively.
1. ENSURE model checkpoint(s) (e.g., `model.safetensors` or `.pt` or `.pth`) is(are) loading and expected files are getting generated from Q-agent and A-agent, when inference is done.
   1. Outputs must be saved to `outputs/questions.json` and `outputs/answers.json`, respectively.

You can test your submission by running the commands in the [Getting Started](#getting-started) section.


## Restrictions
<!-- ⚠️ -->

1.  **<span style="color: red">NO</span> LAST Minute Submission**: The submission deadline is strict. Any changes to your code after the deadline may disqualify your submission.
1.  RAG (Retrieval Augmented Generation) techniques are not allowed.
1.  Adversarial approaches will lead to disqualification, e.g. making A-agents hallucinate.
1.  Only English language is allowed for both Q-agent and A-agent.
1.  Strictly stay within the `max_tokens` limits specified in `agen.yaml` & `qgen.yaml`. Other parameters can be changed.
1.  Questions must pertain to the topics listed in `topics.json`.
1.  Each question should be generated under `10 secs`. Questions exceeding this limit will not be considered.
1.  Each answer should be generated under `6 secs`. Answers exceeding this limit will not be considered.

Feel free to reach out in the Discord channel for any clarifications or questions!

## Directory & Files overview
<!-- 📂  -->

```plaintext
.
├── agents
│   ├── question_model.py
│   ├── question_agent.py
│   ├── answer_model.py
│   └── answer_agent.py
├── assets
│   ├── topics_example.json # example questions w.r.t each topic
│   ├── topics.json # Topics on which we require to generate questions
│   ├── sample_question.json # File specifying expected format of questions generated
│   └── sample_answer.json # Expected format of answers generated
├── utils
│   └── build_prompt.py # prompt-tuning scripts
├── README.ipynb
├── tutorial.ipynb # Synthetic Data Generation and Unsloth Tutorial
├── tutorial_config.yaml # Config file for tutorial
├── qgen.yaml # Generation specific parameters for Q-agent
├── agen.yaml # Generation specific parameters for A-agent
└── default_requirements.txt # Required packages
```
   

## Getting started
<!-- 🎮  -->
Let's get started with running the Q-agent and A-agent framework.

### Environment Setup
<!-- 🚀 -->


```python
# Install the necessary packages
!pip install -r default_requirements.txt
```

    Collecting trl==0.19.0 (from -r default_requirements.txt (line 1))
      Downloading trl-0.19.0-py3-none-any.whl.metadata (10 kB)
    Collecting wandb==0.20.1 (from -r default_requirements.txt (line 2))
      Downloading wandb-0.20.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (10 kB)
    Collecting ipdb==0.13.13 (from -r default_requirements.txt (line 3))
      Downloading ipdb-0.13.13-py3-none-any.whl.metadata (14 kB)
    Collecting transformers==4.51.3 (from -r default_requirements.txt (line 4))
      Downloading transformers-4.51.3-py3-none-any.whl.metadata (38 kB)
    Requirement already satisfied: accelerate>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from trl==0.19.0->-r default_requirements.txt (line 1)) (1.11.0)
    Requirement already satisfied: datasets>=3.0.0 in /usr/local/lib/python3.12/dist-packages (from trl==0.19.0->-r default_requirements.txt (line 1)) (4.3.0)
    Requirement already satisfied: click!=8.0.0,>=7.1 in /usr/local/lib/python3.12/dist-packages (from wandb==0.20.1->-r default_requirements.txt (line 2)) (8.3.0)
    Collecting gitpython!=3.1.29,>=1.0.0 (from wandb==0.20.1->-r default_requirements.txt (line 2))
      Downloading gitpython-3.1.45-py3-none-any.whl.metadata (13 kB)
    Requirement already satisfied: packaging in /usr/local/lib/python3.12/dist-packages (from wandb==0.20.1->-r default_requirements.txt (line 2)) (25.0)
    Requirement already satisfied: platformdirs in /usr/local/lib/python3.12/dist-packages (from wandb==0.20.1->-r default_requirements.txt (line 2)) (4.5.0)
    Requirement already satisfied: protobuf!=4.21.0,!=5.28.0,<7,>=3.19.0 in /usr/local/lib/python3.12/dist-packages (from wandb==0.20.1->-r default_requirements.txt (line 2)) (6.33.0)
    Requirement already satisfied: psutil>=5.0.0 in /usr/local/lib/python3.12/dist-packages (from wandb==0.20.1->-r default_requirements.txt (line 2)) (7.1.1)
    Requirement already satisfied: pydantic<3 in /usr/local/lib/python3.12/dist-packages (from wandb==0.20.1->-r default_requirements.txt (line 2)) (2.12.3)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.12/dist-packages (from wandb==0.20.1->-r default_requirements.txt (line 2)) (6.0.3)
    Requirement already satisfied: requests<3,>=2.0.0 in /usr/local/lib/python3.12/dist-packages (from wandb==0.20.1->-r default_requirements.txt (line 2)) (2.32.5)
    Requirement already satisfied: sentry-sdk>=2.0.0 in /usr/local/lib/python3.12/dist-packages (from wandb==0.20.1->-r default_requirements.txt (line 2)) (2.42.1)
    Requirement already satisfied: setproctitle in /usr/local/lib/python3.12/dist-packages (from wandb==0.20.1->-r default_requirements.txt (line 2)) (1.3.7)
    Requirement already satisfied: typing-extensions<5,>=4.8 in /usr/local/lib/python3.12/dist-packages (from wandb==0.20.1->-r default_requirements.txt (line 2)) (4.15.0)
    Requirement already satisfied: ipython>=7.31.1 in /usr/local/lib/python3.12/dist-packages (from ipdb==0.13.13->-r default_requirements.txt (line 3)) (9.6.0)
    Requirement already satisfied: decorator in /usr/local/lib/python3.12/dist-packages (from ipdb==0.13.13->-r default_requirements.txt (line 3)) (5.2.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from transformers==4.51.3->-r default_requirements.txt (line 4)) (3.20.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.30.0 in /usr/local/lib/python3.12/dist-packages (from transformers==4.51.3->-r default_requirements.txt (line 4)) (0.36.0)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.12/dist-packages (from transformers==4.51.3->-r default_requirements.txt (line 4)) (2.2.0)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from transformers==4.51.3->-r default_requirements.txt (line 4)) (2025.10.23)
    Collecting tokenizers<0.22,>=0.21 (from transformers==4.51.3->-r default_requirements.txt (line 4))
      Downloading tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
    Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/dist-packages (from transformers==4.51.3->-r default_requirements.txt (line 4)) (0.6.2)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.12/dist-packages (from transformers==4.51.3->-r default_requirements.txt (line 4)) (4.67.1)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<1.0,>=0.30.0->transformers==4.51.3->-r default_requirements.txt (line 4)) (2025.9.0)
    Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<1.0,>=0.30.0->transformers==4.51.3->-r default_requirements.txt (line 4)) (1.1.10)
    Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.12/dist-packages (from pydantic<3->wandb==0.20.1->-r default_requirements.txt (line 2)) (0.7.0)
    Requirement already satisfied: pydantic-core==2.41.4 in /usr/local/lib/python3.12/dist-packages (from pydantic<3->wandb==0.20.1->-r default_requirements.txt (line 2)) (2.41.4)
    Requirement already satisfied: typing-inspection>=0.4.2 in /usr/local/lib/python3.12/dist-packages (from pydantic<3->wandb==0.20.1->-r default_requirements.txt (line 2)) (0.4.2)
    Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.0.0->wandb==0.20.1->-r default_requirements.txt (line 2)) (3.4.4)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.0.0->wandb==0.20.1->-r default_requirements.txt (line 2)) (3.11)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.0.0->wandb==0.20.1->-r default_requirements.txt (line 2)) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.0.0->wandb==0.20.1->-r default_requirements.txt (line 2)) (2025.10.5)
    Requirement already satisfied: torch>=2.0.0 in /usr/local/lib/python3.12/dist-packages (from accelerate>=1.4.0->trl==0.19.0->-r default_requirements.txt (line 1)) (2.9.0a0+git1c57644)
    Requirement already satisfied: pyarrow>=21.0.0 in /usr/local/lib/python3.12/dist-packages (from datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (21.0.0)
    Requirement already satisfied: dill<0.4.1,>=0.3.0 in /usr/local/lib/python3.12/dist-packages (from datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (0.4.0)
    Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (from datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (2.3.3)
    Requirement already satisfied: httpx<1.0.0 in /usr/local/lib/python3.12/dist-packages (from datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (0.28.1)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.12/dist-packages (from datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (3.6.0)
    Requirement already satisfied: multiprocess<0.70.17 in /usr/local/lib/python3.12/dist-packages (from datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (0.70.16)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /usr/local/lib/python3.12/dist-packages (from fsspec[http]<=2025.9.0,>=2023.1.0->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (3.13.1)
    Requirement already satisfied: anyio in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (4.11.0)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (1.0.9)
    Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1.0.0->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (0.16.0)
    Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (2.6.1)
    Requirement already satisfied: aiosignal>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (1.4.0)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (25.4.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (1.8.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (6.7.0)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (0.4.1)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (1.22.0)
    Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.29,>=1.0.0->wandb==0.20.1->-r default_requirements.txt (line 2))
      Downloading gitdb-4.0.12-py3-none-any.whl.metadata (1.2 kB)
    Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.29,>=1.0.0->wandb==0.20.1->-r default_requirements.txt (line 2))
      Downloading smmap-5.0.2-py3-none-any.whl.metadata (4.3 kB)
    Requirement already satisfied: ipython-pygments-lexers in /usr/local/lib/python3.12/dist-packages (from ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (1.1.1)
    Requirement already satisfied: jedi>=0.16 in /usr/local/lib/python3.12/dist-packages (from ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (0.19.2)
    Requirement already satisfied: matplotlib-inline in /usr/local/lib/python3.12/dist-packages (from ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (0.2.1)
    Requirement already satisfied: pexpect>4.3 in /usr/local/lib/python3.12/dist-packages (from ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (4.9.0)
    Requirement already satisfied: prompt_toolkit<3.1.0,>=3.0.41 in /usr/local/lib/python3.12/dist-packages (from ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (3.0.52)
    Requirement already satisfied: pygments>=2.4.0 in /usr/local/lib/python3.12/dist-packages (from ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (2.19.2)
    Requirement already satisfied: stack_data in /usr/local/lib/python3.12/dist-packages (from ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (0.6.3)
    Requirement already satisfied: traitlets>=5.13.0 in /usr/local/lib/python3.12/dist-packages (from ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (5.14.3)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.12/dist-packages (from prompt_toolkit<3.1.0,>=3.0.41->ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (0.2.14)
    Requirement already satisfied: parso<0.9.0,>=0.8.4 in /usr/local/lib/python3.12/dist-packages (from jedi>=0.16->ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (0.8.5)
    Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.12/dist-packages (from pexpect>4.3->ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (0.7.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch>=2.0.0->accelerate>=1.4.0->trl==0.19.0->-r default_requirements.txt (line 1)) (79.0.1)
    Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=2.0.0->accelerate>=1.4.0->trl==0.19.0->-r default_requirements.txt (line 1)) (1.14.0)
    Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch>=2.0.0->accelerate>=1.4.0->trl==0.19.0->-r default_requirements.txt (line 1)) (3.5)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=2.0.0->accelerate>=1.4.0->trl==0.19.0->-r default_requirements.txt (line 1)) (3.1.6)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=2.0.0->accelerate>=1.4.0->trl==0.19.0->-r default_requirements.txt (line 1)) (1.3.0)
    Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.12/dist-packages (from anyio->httpx<1.0.0->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (1.3.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=2.0.0->accelerate>=1.4.0->trl==0.19.0->-r default_requirements.txt (line 1)) (3.0.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (2025.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas->datasets>=3.0.0->trl==0.19.0->-r default_requirements.txt (line 1)) (1.17.0)
    Requirement already satisfied: executing>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from stack_data->ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (2.2.1)
    Requirement already satisfied: asttokens>=2.1.0 in /usr/local/lib/python3.12/dist-packages (from stack_data->ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (3.0.0)
    Requirement already satisfied: pure-eval in /usr/local/lib/python3.12/dist-packages (from stack_data->ipython>=7.31.1->ipdb==0.13.13->-r default_requirements.txt (line 3)) (0.2.3)
    Downloading trl-0.19.0-py3-none-any.whl (375 kB)
    Downloading wandb-0.20.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (23.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m23.2/23.2 MB[0m [31m111.5 MB/s[0m  [33m0:00:00[0mm0:00:01[0m
    [?25hDownloading ipdb-0.13.13-py3-none-any.whl (12 kB)
    Downloading transformers-4.51.3-py3-none-any.whl (10.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.4/10.4 MB[0m [31m129.4 MB/s[0m  [33m0:00:00[0m
    [?25hDownloading tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.1/3.1 MB[0m [31m231.7 MB/s[0m  [33m0:00:00[0m
    [?25hDownloading gitpython-3.1.45-py3-none-any.whl (208 kB)
    Downloading gitdb-4.0.12-py3-none-any.whl (62 kB)
    Downloading smmap-5.0.2-py3-none-any.whl (24 kB)
    Installing collected packages: smmap, gitdb, tokenizers, ipdb, gitpython, wandb, transformers, trl
    [2K  Attempting uninstall: tokenizers
    [2K    Found existing installation: tokenizers 0.22.1
    [2K    Uninstalling tokenizers-0.22.1:
    [2K      Successfully uninstalled tokenizers-0.22.1
    [2K  Attempting uninstall: transformersm[90m╺[0m[90m━━━━━━━━━━━━━━[0m [32m5/8[0m [wandb]hon]
    [2K    Found existing installation: transformers 4.56.2━━━━━━━━━━[0m [32m5/8[0m [wandb]
    [2K    Uninstalling transformers-4.56.2:━[0m[90m╺[0m[90m━━━━━━━━━[0m [32m6/8[0m [transformers]
    [2K      Successfully uninstalled transformers-4.56.20m[90m━━━━━━━━━[0m [32m6/8[0m [transformers]
    [2K  Attempting uninstall: trl━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━[0m [32m6/8[0m [transformers]
    [2K    Found existing installation: trl 0.23.0╺[0m[90m━━━━━━━━━[0m [32m6/8[0m [transformers]
    [2K    Uninstalling trl-0.23.0:━━━━━━[0m[90m╺[0m[90m━━━━━━━━━[0m [32m6/8[0m [transformers]
    [2K      Successfully uninstalled trl-0.23.0━━[0m[90m╺[0m[90m━━━━[0m [32m7/8[0m [trl]mers]
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m8/8[0m [trl]m7/8[0m [trl]
    [1A[2K[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    unsloth 2025.10.11 requires xformers>=0.0.27.post2; ("linux" in sys_platform or sys_platform == "win32") and (platform_machine == "AMD64" or platform_machine == "x86_64"), which is not installed.
    unsloth-zoo 2025.10.12 requires cut_cross_entropy; python_version >= "3.10", which is not installed.
    unsloth-zoo 2025.10.12 requires torchao>=0.13.0, which is not installed.
    unsloth 2025.10.11 requires trl!=0.19.0,<=0.23.0,>=0.18.2, but you have trl 0.19.0 which is incompatible.
    unsloth-zoo 2025.10.12 requires trl!=0.19.0,<=0.23.0,>=0.18.2, but you have trl 0.19.0 which is incompatible.
    vllm 0.11.1rc3.dev39+gf417746ad.rocm700 requires transformers>=4.56.0, but you have transformers 4.51.3 which is incompatible.[0m[31m
    [0mSuccessfully installed gitdb-4.0.12 gitpython-3.1.45 ipdb-0.13.13 smmap-5.0.2 tokenizers-0.21.4 transformers-4.51.3 trl-0.19.0 wandb-0.20.1
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.[0m[33m
    [0m


```python
# Import basic packages
import json
from typing import Dict, Any, List
```

### Q-Agent
<!-- 🤔 -->
You will update the model in `question_model.py`, which will be invoked by `question_agent.py`. In the provided skeleton, we have used the base Qwen3-4B model for Q-Agent but you should experiment with other models and techniques. Check out our [Synthetic Data Generation and Unsloth Tutorial](./tutorial.ipynb) for training tips and tricks.

Generated questions must pertain to the topics mentioned in `topics.json` file. Additional topics will be added for the tournament finals.

__Topics:__
1.  `Puzzles`: Seating Arrangements (Linear, Circular)
2.  `Blood Relations and Family Tree`: Puzzles involving generations and family tree logic

Sample questions and answers are available in the [assets folder](./assets).


```python
# Run the following code to generate questions.
# For demo purpose, we have used the base Qwen3-4B model for Q-Agent. Participants are expected to improve upon this
!python -m agents.question_agent \
    --output_file "outputs/questions.json" \
    --num_questions 20 \
    --verbose
```

    Loading checkpoint shards: 100%|████████████████| 30/30 [00:54<00:00,  1.81s/it]
    STEPS: 100%|██████████████████████████████████████| 4/4 [02:28<00:00, 37.01s/it]
    Generated 20 questions!
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?",
      "choices": ["A) Alex", "B) Ben", "C) Charlie", "D) David"],
      "answer": "A",
      "explanation": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Eliminate impossible configurations. Step 4: Conclude Alex drinks coffee."
    }
    {
      "topic": "Puzzles involving generations and family tree logic",
      "question": "Rohan's paternal grandfather has three sons and two daughters. The eldest son's only daughter is married to the youngest son of Rohan's maternal grandmother's only brother. How is Rohan related to the eldest son's daughter's husband?",
      "choices": ["A) Brother-in-law", "B) Nephew", "C) Uncle", "D) Cousin"],
      "answer": "D",
      "explanation": "Step 1: Identify Rohan's relatives. Step 2: The eldest son's daughter is Rohan's cousin. Step 3: Her husband is the youngest son of Rohan's maternal grandmother's brother, making him Rohan's cousin through marriage."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits two seats to the left of Ben. Charlie sits opposite Diana. Eve sits two seats to the right of Frank. George sits adjacent to both Alex and Charlie. Who sits between Ben and Diana?",
      "choices": ["A) Eve", "B) Frank", "C) George", "D) Charlie"],
      "answer": "D",
      "explanation": "Step 1: Place Alex and Ben. Step 2: Position Charlie opposite Diana. Step 3: Place Eve and Frank relative to each other. Step 4: George's position is between Alex and Charlie, which means Charlie must be between Ben and Diana to satisfy all conditions."
    }
    {
      "topic": "Puzzles involving generations and family tree logic",
      "question": "Rahul's mother's only brother is married to the sister of the father of the wife of Rahul's paternal uncle. How is Rahul related to the brother of the wife of Rahul's paternal uncle?",
      "choices": ["A) Brother-in-law", "B) Nephew", "C) Cousin", "D) Uncle"],
      "answer": "C",
      "explanation": "Step 1: Rahul's mother's only brother is Rahul's maternal uncle. Step 2: The sister of the father of the wife of Rahul's paternal uncle is the mother of Rahul's paternal uncle's wife. Step 3: The brother of the wife of Rahul's paternal uncle is the cousin of Rahul's paternal uncle's wife. Step 4: Since Rahul's paternal uncle is Rahul's father's brother, Rahul's paternal uncle's wife is Rahul's aunt. Step 5: The cousin of Rahul's aunt is Rahul's cousin. Therefore, Rahul is the cousin of the brother of the wife of Rahul's paternal uncle."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?",
      "choices": ["A) Alex", "B) Charlie", "C) David", "D) Emily"],
      "answer": "B",
      "explanation": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Apply the constraints to determine the coffee drinker."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?",
      "choices": ["A) Alex", "B) Ben", "C) Charlie", "D) David"],
      "answer": "A",
      "explanation": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Eliminate impossible configurations. Step 4: Conclude Alex drinks coffee."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?",
      "choices": ["A) Alex", "B) Ben", "C) Charlie", "D) David"],
      "answer": "D",
      "explanation": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Apply the constraints to determine the coffee drinker's identity."
    }
    {
      "topic": "Puzzles involving generations and family tree logic",
      "question": "Rohan's paternal grandfather has three sons and two daughters. The eldest son's only daughter is married to the youngest son of Rohan's maternal grandmother's only brother. How is Rohan related to the eldest son's daughter's husband?",
      "choices": ["A) Brother-in-law", "B) Nephew", "C) Uncle", "D) Cousin"],
      "answer": "D",
      "explanation": "Step 1: Identify Rohan's relatives. Step 2: The eldest son's daughter is Rohan's cousin. Step 3: Her husband is the youngest son of Rohan's maternal grandmother's brother, making him Rohan's cousin through marriage."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits two seats to the left of Ben. Charlie sits opposite Diana. Emily sits two seats to the right of Frank. George sits adjacent to both Alex and Charlie. Who sits between Ben and Diana?",
      "choices": ["A) Emily", "B) Frank", "C) George", "D) Alex"],
      "answer": "B",
      "explanation": "Step 1: Place Alex two seats left of Ben. Step 2: Position Charlie opposite Diana. Step 3: Place Emily two seats right of Frank. Step 4: George sits adjacent to both Alex and Charlie, implying George is between Alex and Charlie. Step 5: Since Charlie is opposite Diana, and Alex is two seats left of Ben, Frank must sit between Ben and Diana to satisfy the circular arrangement and given constraints."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?",
      "choices": ["A) Alex", "B) Ben", "C) Charlie", "D) David"],
      "answer": "A",
      "explanation": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Apply the constraints to determine the coffee drinker's identity."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?",
      "choices": ["A) Alex", "B) Ben", "C) Charlie", "D) David"],
      "answer": "A",
      "explanation": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Eliminate impossible configurations. Step 4: Conclude Alex drinks coffee."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?",
      "choices": ["A) Alex", "B) Ben", "C) Charlie", "D) David"],
      "answer": "A",
      "explanation": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Eliminate impossible configurations. Step 4: Conclude Alex drinks coffee."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits two seats to the left of Ben. Charlie sits opposite Diana. Eve sits two seats to the right of Frank. George sits adjacent to both Alex and Charlie. Who sits between Ben and Diana?",
      "choices": ["A) Eve", "B) Frank", "C) George", "D) Charlie"],
      "answer": "D",
      "explanation": "Step 1: Place Alex and Ben. Step 2: Position Charlie opposite Diana. Step 3: Place Eve and Frank relative to each other. Step 4: George's position is between Alex and Charlie, which means Charlie must be between Ben and Diana to satisfy all conditions."
    }
    {
      "topic": "Puzzles involving generations and family tree logic",
      "question": "Rohan's mother's only brother is married to the sister of Rohan's father's only daughter. How is Rohan related to the brother's wife?",
      "choices": ["A) Nephew", "B) Brother-in-law", "C) Uncle", "D) Father"],
      "answer": "A",
      "explanation": "Step 1: Identify Rohan's mother's only brother as Rohan's maternal uncle. Step 2: Recognize the sister of Rohan's father's only daughter as Rohan's sister. Step 3: Since the maternal uncle is married to Rohan's sister, this makes the brother's wife Rohan's sister, and thus Rohan is her brother, but more specifically in the context of the question, the relationship asked is from the wife to Rohan, making Rohan her nephew."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?",
      "choices": ["A) Alex", "B) Ben", "C) Charlie", "D) David"],
      "answer": "C",
      "explanation": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Apply the constraints to determine the coffee drinker."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?",
      "choices": ["A) Alex", "B) Ben", "C) Charlie", "D) David"],
      "answer": "A",
      "explanation": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Eliminate impossible configurations. Step 4: Conclude Alex drinks coffee."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits two seats to the left of Ben. Charlie sits opposite Diana. Emily sits two seats to the right of Frank. George sits adjacent to both Alex and Charlie. Who sits between Ben and Diana?",
      "choices": ["A) Emily", "B) Frank", "C) George", "D) Alex"],
      "answer": "C",
      "explanation": "Step 1: Place Alex two seats left of Ben. Step 2: Position Charlie opposite Diana. Step 3: Place Emily two seats right of Frank. Step 4: George sits adjacent to both Alex and Charlie, fitting between Ben and Diana."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits two seats to the left of Ben. Charlie sits opposite Diana. Emily sits two seats to the right of Frank. George sits adjacent to both Alex and Charlie. Who sits between Ben and Diana?",
      "choices": ["A) Emily", "B) Frank", "C) George", "D) Alex"],
      "answer": "C",
      "explanation": "Step 1: Place Alex two seats left of Ben. Step 2: Position Charlie opposite Diana. Step 3: Place Emily two seats right of Frank. Step 4: George sits adjacent to both Alex and Charlie, fitting between Ben and Diana."
    }
    {
      "topic": "Seating Arrangements (Linear, Circular)",
      "question": "Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?",
      "choices": ["A) Alex", "B) Ben", "C) Charlie", "D) David"],
      "answer": "D",
      "explanation": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Apply the constraints to determine the coffee drinker's identity."
    }
    {
      "topic": "Puzzles involving generations and family tree logic",
      "question": "Rahul's mother's only brother is married to the sister of the father of the wife of Rahul's paternal uncle. How is Rahul related to the brother of the wife of Rahul's paternal uncle?",
      "choices": ["A) Brother-in-law", "B) Nephew", "C) Cousin", "D) Uncle"],
      "answer": "C",
      "explanation": "Step 1: Rahul's mother's only brother is Rahul's maternal uncle. Step 2: The sister of the father of the wife of Rahul's paternal uncle is the mother of Rahul's paternal uncle's wife. Step 3: The brother of the wife of Rahul's paternal uncle is the cousin of Rahul's paternal uncle's wife. Step 4: Since Rahul's paternal uncle is Rahul's father's brother, Rahul's paternal uncle's wife is Rahul's aunt. Step 5: The cousin of Rahul's aunt is Rahul's cousin. Therefore, Rahul is the cousin of the brother of the wife of Rahul's paternal uncle."
    }
    
    ==================================================
    
    
    Time taken per batch generation: [55.79134488105774, 28.952353715896606, 29.250231504440308, 33.97560906410217]
    Tokens generated per batch: [1175, 985, 995, 1175]
    Total Time Taken: 147.970 seconds; Total Tokens: 4330; TGPS: 29.263 seconds
    
    
    
    ++++++++++++++++++++++++++++++++++++++++++++++++++
    
    Saved to outputs/questions.json!


#### Basic format-checks for questions from Q-agent

Generated questions must follow the [format instructions](#format-overview). All questions generated from the Q-agent will be filtered and validated before being sent to the opponent's A-agent. We generate two version of questions, one is the raw, unfiltered one `questions.json` and the other is `filtered_questions.json` after passing through the below example filter. The full filtering and validation process is part of the judging system and is not demonstrated here.



```python
from transformers import AutoTokenizer
from typing import List, Dict, Any
tokenizer = AutoTokenizer.from_pretrained("/workspace/AIAC/logical_reasoning/logical_reasoning_rocm_merged", padding_side='left')

def count_tokens_q(text: str) -> int:
    """Count the number of tokens using Qwen3-4B tokenizer"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def filter_questions(questions: List[str|Dict[str, str|Any]]) -> List[Dict[str, str|Any]]:
    def basic_checks(q2: Dict[str, str])->bool:
        # check required keys
        required_keys = ['topic', 'question', 'choices', 'answer']
        if all((key in q2) for key in required_keys):
            # check choices format
            checks = all(isinstance(choice, str) and len(choice) > 2 and choice[0].upper() in 'ABCD' for choice in q2['choices'])
            if isinstance(q2['choices'], list) and len(q2['choices']) == 4 and checks:
                # check answer format
                # Check token length
                check_len = sum(count_tokens_q(q2[k]) for k in ['question', 'answer'])
                check_len += sum(count_tokens_q(choice) for choice in q2['choices']) - 15
                if check_len < 130:
                    if check_len + count_tokens_q(q2.get('explanation', 'None')) <= 1024:
                        # Extra Checks: (PLUS checks) len(q2['answer']) == 1 and q2['answer'].upper() in 'ABCD':
                        if isinstance(q2['answer'], str):
                            return True
        return False
    correct_format_question = []
    for i, q in enumerate(questions):
        if isinstance(q, dict):
            if basic_checks(q):
                correct_format_question.append(q)
        elif isinstance(q, str):
            try:
                q1 = json.loads(q)
                if basic_checks(q1):
                    correct_format_question.append(q1)
            except json.JSONDecodeError:
                # If JSON decoding fails, skip this answer
                print(f"Skipping invalid JSON at index {i}: {q}")
                continue
        else:
            continue
    if len(correct_format_question) >= 0.5 * len(questions):
        return correct_format_question
    return list()
```


```python
import json
with open("outputs/questions.json", "r") as f:
    questions = json.load(f)

filtered_questions = filter_questions(questions)

with open("outputs/filtered_questions.json", "w") as f:
    json.dump(filtered_questions, f, indent=4)

print(f"Number of questions: {len(questions)}")
print(f"Number of filtered questions: {len(filtered_questions)}")
```

    Number of questions: 20
    Number of filtered questions: 20


### A-agent
<!-- 🤖  -->
You will update the model in `answer_model.py`, which will be invoked by `answer_agent.py`. In the provided skeleton, we have again used the base Qwen3-4B model for A-Agent but you should experiment with other models and techniques. Check out our [Synthetic Data Generation and Unsloth Tutorial](./tutorial.ipynb) for training tips and tricks.


```python
# Same instructions apply for the answer agent.
# For demo purpose, we have used the base Qwen3-4B model for A-agent. Participants are expected to improve upon this.
!python -m agents.answer_agent \
    --input_file "outputs/filtered_questions.json" \
    --output_file "outputs/answers.json" \
    --verbose
```

    Loading checkpoint shards: 100%|████████████████| 30/30 [00:53<00:00,  1.77s/it]
    STEPS: 100%|███████████████████████████████████| 4/4 [01:06<00:00, 16.53s/batch]
    
    === Question 1 ===
    Question: Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?
    Expected: A
    Model Answer:
    {"answer": "A", "reasoning": "Through systematic elimination and analysis of seating constraints, Alex is determined to drink coffee."}
    
    === Question 2 ===
    Question: Rohan's paternal grandfather has three sons and two daughters. The eldest son's only daughter is married to the youngest son of Rohan's maternal grandmother's only brother. How is Rohan related to the eldest son's daughter's husband?
    Expected: D
    Model Answer:
    {"answer": "D", "reasoning": "Step 1: Identify Rohan's relatives. Step 2: The eldest son's daughter is Rohan's cousin. Step 3: Her husband is the youngest son of Rohan's maternal grandmother's brother, making him Rohan's cousin through marriage. Step 4: Since the husband is the cousin of Rohan's cousin, he is also Rohan's cousin. Step 5: Therefore, the correct answer is D) Cousin."}
    
    === Question 3 ===
    Question: Seven friends sit in a circle. Alex sits two seats to the left of Ben. Charlie sits opposite Diana. Eve sits two seats to the right of Frank. George sits adjacent to both Alex and Charlie. Who sits between Ben and Diana?
    Expected: D
    Model Answer:
    {"answer": "D", "reasoning": "Step 1: Place Alex and Ben. Step 2: Position Charlie opposite Diana. Step 3: Place Eve and Frank relative to each other. Step 4: George's position is between Alex and Charlie, which means Charlie must be between Ben and Diana to satisfy all conditions."}
    
    === Question 4 ===
    Question: Rahul's mother's only brother is married to the sister of the father of the wife of Rahul's paternal uncle. How is Rahul related to the brother of the wife of Rahul's paternal uncle?
    Expected: C
    Model Answer:
    {"answer": "C", "reasoning": "Step 1: Identify Rahul's relationships to the individuals mentioned. Step 2: Determine the brother of the wife of Rahul's paternal uncle's relationship to Rahul. Step 3: Conclude based on the explanation provided that Rahul is the cousin of the brother of the wife of Rahul's paternal uncle, making option C the correct choice."}
    
    === Question 5 ===
    Question: Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?
    Expected: B
    Model Answer:
    {"answer": "C", "reasoning": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Apply the constraints to determine the coffee drinker. Step 4: Since Alex sits opposite the person who drinks tea, and Ben sits two seats to the left of the coffee drinker, we can deduce the coffee drinker's position. Step 5: Charlie drinks juice, and milk is drunk by someone adjacent to juice, which helps to further constrain the possible positions. Step 6: Through systematic elimination of invalid configurations, we find that David is the only possible coffee drinker."}
    
    === Question 6 ===
    Question: Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?
    Expected: A
    Model Answer:
    {"answer": "A", "reasoning": "Through systematic elimination and analysis of seating constraints, Alex is determined to drink coffee."}
    
    === Question 7 ===
    Question: Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?
    Expected: D
    Model Answer:
    {"answer": "B", "reasoning": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Apply the constraints to determine the coffee drinker's identity. Step 4: Since Alex sits opposite the person who drinks tea, and Ben sits two seats to the left of the coffee drinker, we can deduce that Ben is the coffee drinker. Step 5: Charlie drinks juice, and milk is drunk by someone adjacent to juice, which does not affect the coffee drinker's position. Step 6: Therefore, through logical deduction, we conclude that Ben drinks coffee."}
    
    === Question 8 ===
    Question: Rohan's paternal grandfather has three sons and two daughters. The eldest son's only daughter is married to the youngest son of Rohan's maternal grandmother's only brother. How is Rohan related to the eldest son's daughter's husband?
    Expected: D
    Model Answer:
    {"answer": "D", "reasoning": "Step 1: Identify Rohan's relatives. Step 2: The eldest son's daughter is Rohan's cousin. Step 3: Her husband is the youngest son of Rohan's maternal grandmother's brother, making him Rohan's cousin through marriage. Step 4: Since the husband is the cousin of Rohan's cousin, he is also Rohan's cousin. Step 5: Therefore, the correct answer is D) Cousin."}
    
    === Question 9 ===
    Question: Seven friends sit in a circle. Alex sits two seats to the left of Ben. Charlie sits opposite Diana. Emily sits two seats to the right of Frank. George sits adjacent to both Alex and Charlie. Who sits between Ben and Diana?
    Expected: B
    Model Answer:
    {"answer": "B", "reasoning": "Step 1: Place Alex two seats left of Ben. Step 2: Position Charlie opposite Diana. Step 3: Place Emily two seats right of Frank. Step 4: George sits adjacent to both Alex and Charlie, implying George is between Alex and Charlie. Step 5: Since Charlie is opposite Diana, and Alex is two seats left of Ben, Frank must sit between Ben and Diana to satisfy the circular arrangement and given constraints."}
    
    === Question 10 ===
    Question: Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?
    Expected: A
    Model Answer:
    {"answer": "B", "reasoning": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Apply the constraints to determine the coffee drinker's identity. Step 4: Since Alex sits opposite the person who drinks tea, and Ben sits two seats to the left of the coffee drinker, we can deduce that Ben is the coffee drinker. Step 5: Charlie drinks juice, and milk is drunk by someone adjacent to juice, which does not affect the coffee drinker's position. Step 6: Therefore, through logical deduction, we conclude that Ben drinks coffee."}
    
    === Question 11 ===
    Question: Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?
    Expected: A
    Model Answer:
    {"answer": "A", "reasoning": "Through systematic elimination and analysis of seating constraints, Alex is determined to drink coffee."}
    
    === Question 12 ===
    Question: Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?
    Expected: A
    Model Answer:
    {"answer": "A", "reasoning": "Through systematic elimination and analysis of seating constraints, Alex is determined to drink coffee."}
    
    === Question 13 ===
    Question: Seven friends sit in a circle. Alex sits two seats to the left of Ben. Charlie sits opposite Diana. Eve sits two seats to the right of Frank. George sits adjacent to both Alex and Charlie. Who sits between Ben and Diana?
    Expected: D
    Model Answer:
    {"answer": "D", "reasoning": "Step 1: Place Alex and Ben. Step 2: Position Charlie opposite Diana. Step 3: Place Eve and Frank relative to each other. Step 4: George's position is between Alex and Charlie, which means Charlie must be between Ben and Diana to satisfy all conditions."}
    
    === Question 14 ===
    Question: Rohan's mother's only brother is married to the sister of Rohan's father's only daughter. How is Rohan related to the brother's wife?
    Expected: A
    Model Answer:
    {"answer": "A", "reasoning": "Rohan is the nephew of his maternal uncle's wife, as she is his sister, making option A the correct choice through logical family relationship analysis."}
    
    === Question 15 ===
    Question: Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?
    Expected: C
    Model Answer:
    {"answer": "B", "reasoning": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Apply the constraints to determine the coffee drinker. Step 4: Since Alex sits opposite the person who drinks tea, and Ben sits two seats to the left of the coffee drinker, we can deduce that Ben does not drink tea. Step 5: Charlie drinks juice, and milk is drunk by someone adjacent to juice. Step 6: David is adjacent to Emily, but there is no direct constraint linking them to the coffee drinker. Step 7: Considering the constraints, Ben is the only one who can be two seats to the left of the coffee drinker without violating any conditions. Step 8: Therefore, the coffee drinker must be two seats to the right of Ben, making option B the correct choice."}
    
    === Question 16 ===
    Question: Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?
    Expected: A
    Model Answer:
    {"answer": "A", "reasoning": "Through systematic elimination and analysis of seating constraints, Alex is determined to drink coffee."}
    
    === Question 17 ===
    Question: Seven friends sit in a circle. Alex sits two seats to the left of Ben. Charlie sits opposite Diana. Emily sits two seats to the right of Frank. George sits adjacent to both Alex and Charlie. Who sits between Ben and Diana?
    Expected: C
    Model Answer:
    {"answer": "C", "reasoning": "Step 1: Place Alex two seats left of Ben. Step 2: Position Charlie opposite Diana. Step 3: Place Emily two seats right of Frank. Step 4: George sits adjacent to both Alex and Charlie, fitting between Ben and Diana. Step 5: Through systematic elimination and arrangement, George is the only one who can sit between Ben and Diana while satisfying all conditions."}
    
    === Question 18 ===
    Question: Seven friends sit in a circle. Alex sits two seats to the left of Ben. Charlie sits opposite Diana. Emily sits two seats to the right of Frank. George sits adjacent to both Alex and Charlie. Who sits between Ben and Diana?
    Expected: C
    Model Answer:
    {"answer": "C", "reasoning": "Step 1: Place Alex two seats left of Ben. Step 2: Position Charlie opposite Diana. Step 3: Place Emily two seats right of Frank. Step 4: George sits adjacent to both Alex and Charlie, fitting between Ben and Diana. Step 5: Through systematic elimination and arrangement, George is the only one who can sit between Ben and Diana while satisfying all conditions."}
    
    === Question 19 ===
    Question: Seven friends sit in a circle. Alex sits opposite the person who drinks tea. Ben sits two seats to the left of the coffee drinker. Charlie drinks juice. David is adjacent to Emily. Milk is drunk by someone adjacent to juice. Who drinks coffee?
    Expected: D
    Model Answer:
    {"answer": "B", "reasoning": "Step 1: Analyze the seating constraints. Step 2: Identify the coffee drinker's position relative to Ben. Step 3: Apply the constraints to determine the coffee drinker's identity. Step 4: Since Alex sits opposite the person who drinks tea, and Ben sits two seats to the left of the coffee drinker, we can deduce that Ben cannot be the coffee drinker. Step 5: Charlie drinks juice, and milk is drunk by someone adjacent to juice, so Charlie cannot be the coffee drinker. Step 6: David is adjacent to Emily, but there is no direct constraint linking them to the coffee drinker. Step 7: Through systematic elimination and logical deduction, we find that Ben is indeed the coffee drinker, as it satisfies all given constraints without creating any conflicts."}
    
    === Question 20 ===
    Question: Rahul's mother's only brother is married to the sister of the father of the wife of Rahul's paternal uncle. How is Rahul related to the brother of the wife of Rahul's paternal uncle?
    Expected: C
    Model Answer:
    {"answer": "C", "reasoning": "Step 1: Identify Rahul's relationships to the individuals mentioned. Step 2: Determine the relationship between Rahul's paternal uncle and his wife. Step 3: Analyze the family connections to find the brother of Rahul's paternal uncle's wife. Step 4: Conclude the relationship between Rahul and this individual based on family ties."}
    BATCH - 0
    Tokens: 705, Time: 31.430 seconds
    TGPS: 22.431 seconds
    BATCH - 1
    Tokens: 700, Time: 9.453 seconds
    TGPS: 74.048 seconds
    BATCH - 2
    Tokens: 955, Time: 12.880 seconds
    TGPS: 74.148 seconds
    BATCH - 3
    Tokens: 895, Time: 12.324 seconds
    TGPS: 72.624 seconds
    
    ==================================================
    Total Time: 66.086 seconds; Total Tokens: 3255; TGPS: 49.254 seconds


#### Basic format-checks for answers from A-agent
Generated answers must follow the [format instructions](#format-overview). The following filter is added into the `answer_agent.py`. Similarly here too, two versions are saved, `answers.json` and `filtered_answers.json`. The latter is used for evaluation.


```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/workspace/AIAC/logical_reasoning/logical_reasoning_rocm_merged", padding_side='left')

def count_tokens_a(text: str) -> int:
    """Count the number of tokens in the text using the agent's tokenizer"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def filter_answers(ans: List[str|Dict[str, str]]) -> List[Dict[str, str]]:
    r"""Filter answers to ensure they are in the correct format"""
    def basic_checks(a1: Dict[str, str])->bool:
        # check required keys
        required_keys = ['answer']
        if all((key in a1) and isinstance(a1[key], str) for key in required_keys):
            if len(a1['answer']) == 1 and (a1['answer'] not in 'ABCDabcd'):
                    return False
            check_len = count_tokens_a(a1['answer'])
            if check_len < 50:
                check_len += count_tokens_a(a1.get('reasoning', 'None'))
                if check_len < 512:
                    # check answer format - EXTRA checks
                    # if len(a1['answer']) == 1 and a1['answer'].upper() in 'ABCD':
                    return True
        return False

    filtered_answers = []
    for i, a in enumerate(ans):
        if isinstance(a, dict):
            if basic_checks(a):
                filtered_answers.append(a)
            else:
                filtered_answers.append(None)
        elif isinstance(a, str):
            # Basic checks: at least with correct JSON format
            try:
                a1 = json.loads(a)
                if basic_checks(a1):
                    filtered_answers.append(a1)
                else:
                    filtered_answers.append(None)
            except json.JSONDecodeError:
                # If JSON decoding fails, skip this answer
                print(f"Skipping invalid JSON at index {i}: {a}")
                filtered_answers.append(None)
                continue
        else:
            # If the answer is neither a dict nor a str, skip it
            print(f"Skipping unsupported type at index {i}: {type(a)}")
            filtered_answers.append(None)
    return filtered_answers
```


```python
with open("outputs/answers.json", "r") as f:
    answers = json.load(f)
filtered_answers = filter_answers(answers)


print(f"Number of answers: {len(answers)}")
print(f"Number of filtered answers: {len(filtered_answers)}")
```

    Number of answers: 20
    Number of filtered answers: 20


## Evaluation
<!-- 🏅  -->

### Scoring Criteria

<!-- 📊  -->

Scores are assigned based on: out of $N$ questions from Q-agent, how many an A-agent can answer and vice-versa. *No negative marking for wrong answers.*

$$\text{A-agent Score} = \dfrac{\#\ \text{of questions correctly answered with expected format}}{N}\times 100$$
$$\text{Q-agent Score} = \dfrac{\#\ \text{of questions incorrectly answered by A-agent}}{N}\times 100$$


$N$ denotes the number of filtered / format-correct questions. **Teams whose Q-agent fails to generate at least $50\%$ of `num_questions` (where `num_questions` ranges from $2$ to $1000+$) of the questions correctly (as per [format-checking](#format-overview)) will be automatically disqualified.**<br>

In case of **TIE**, closed benchmark questions will be used to evaluate the answer agents (A-agent) and rank the teams accordingly.


### Scoring Example


```python
# calculate scores...
N = len(filtered_questions)
assert N == len(filtered_answers), "Number of questions and answers must match."
num_correct_answers = len([1 for q,a in zip(filtered_questions, filtered_answers) if a is not None and q['answer'] == a['answer']])

# Here the answer may be correct, but since q['answer'] is not an option letter is not there, we face problems
# Below shown is one way of simple string parsing
num_correct_answers = len([1 for q,a in zip(filtered_questions, filtered_answers) if a is not None and q['answer'][0] == a['answer']])

a_score = num_correct_answers*100/(N+1e-9)
q_score = (N-num_correct_answers)*100/(N+1e-9)
# Announce the scores
print(f"Number of questions: {N}")
print(f"Number of correct answers: {num_correct_answers}")
print("Scores:")
print(f"Team B: A-agent score: {a_score:.2f}")
print(f"Team A: Q-agent score: {q_score:.2f}")
print(f"Innings 1 winner: {'Team A' if q_score > a_score else 'Team B' if q_score < a_score else 'Draw'}")
```

    Number of questions: 20
    Number of correct answers: 15
    Scores:
    Team B: A-agent score: 75.00
    Team A: Q-agent score: 25.00
    Innings 1 winner: Team B
