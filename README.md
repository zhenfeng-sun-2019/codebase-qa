# embed.py 说明

`src/rag/embed.py` 是代码库问答助手（codebase-qa）的第一块积木：把代码变成向量，并计算两个向量有多相似。

## 它做什么

文件里有两个函数和一个实验。

| 名称 | 输入 | 输出 | 谁来算 |
|---|---|---|---|
| `embed(texts)` | 一组字符串 | 一组向量，每个字符串对应一个 | Voyage 的模型（`voyage-code-3`） |
| `cosine_similarity(a, b)` | 两个向量 | 一个数字，越大越像 | 自己用纯 Python 写的公式 |

把文字变成有意义的向量需要训练好的模型，所以交给 Voyage API。相似度只是一个数学公式，自己写几行就够了。

## 运行方法

```bash
source .venv/bin/activate
pip install voyageai python-dotenv
python src/rag/embed.py
```

运行前要在仓库根目录建一个 `.env` 文件，写入 API key：

```
VOYAGE_API_KEY=你的key
```

`.env` 已经写进 `.gitignore`，不会被提交。

## 代码说明

### 初始化

```python
load_dotenv()
client = voyageai.Client()
```

`load_dotenv()` 把 `.env` 里的 key 读进环境变量，`voyageai.Client()` 创建客户端并自动使用这个 key。注意 `Client` 的 C 要大写，后面要加括号。写成 `voyageai.client` 拿到的是一个模块，调用 `embed` 时会报 `AttributeError`。

### embed

```python
def embed(texts: list[str]) -> list[list[float]]:
    result = client.embed(texts, model="voyage-code-3", input_type="document")
    return result.embeddings
```

一次调用可以送进去多段文字，返回同样数量的向量。`voyage-code-3` 是专门为代码训练的模型，每个向量 1024 维。

### cosine_similarity

```python
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)
```

对应的公式：

```
cos(a, b) = (a·b) / (|a| × |b|)

a·b = a1*b1 + a2*b2 + ...       对应位置相乘再求和
|a| = sqrt(a1² + a2² + ...)     向量的长度
```

`zip(a, b)` 把两个列表按位置配对，所以 `sum(x * y for x, y in zip(a, b))` 就是 a·b。

## 为什么用 cosine

cosine 只看向量的方向，不看长度。用两个小例子验证：

| 比较 | 结果 | 说明 |
|---|---|---|
| `(1,1)` 和 `(2,2)` | 1.0 | 长度差一倍，方向相同，判定为完全一样 |
| `(1,1)` 和 `(1,0)` | 0.707 | 夹角 45 度，cos 45° = 0.707 |

同一个函数加了很多注释，文字变长了，但意思没变。只看方向的 cosine 更能反映"意思一样"。

注意区分两个概念：

- **相似度**（similarity）：越大越像，最大 1.0
- **距离**（distance）：越小越像，最小 0

有些工具返回的是"cosine 距离"，等于 `1 - cosine 相似度`，用的时候要确认是哪一种，否则排序会反过来。

## 实验：三段代码的相似度

准备三段代码，都存成字符串（三引号里的内容不会被执行，只是文字）：

- A：冒泡排序，两层循环，7 行
- B：用 `sorted()` 排序，1 行
- C：读取文件

实际运行结果（2026/11/10）：

| 比较 | 相似度 |
|---|---|
| A-B（两种排序） | 0.727 |
| B-C（排序 vs 读文件） | 0.469 |
| A-C（排序 vs 读文件） | 0.425 |

## 从结果里学到的

1. **模型比较的是意思，不是字面。** A 和 B 的写法完全不同，相同的词很少，但都在做排序，所以相似度最高。关键词搜索做不到这一点。
2. **无关的代码相似度也不会接近 0。** 三段都是 Python 短函数，本身就有共同点，所以 A-C 仍有 0.43。
3. **看排序，不看绝对值。** 检索时不要给相似度设固定的及格线，而是取排在最前面的几个（top-k）。

## 下一步

把一个真实的 Python 文件按函数切成块（chunking），对所有块做 embedding，然后用一个问题去找最相关的那一块。
