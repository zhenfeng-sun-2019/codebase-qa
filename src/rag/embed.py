import math

import voyageai
from dotenv import load_dotenv

load_dotenv()
client = voyageai.Client()


def embed(texts: list[str]) -> list[list[float]]:
    result = client.embed(texts, model="voyage-code-3", input_type="document")
    return result.embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

if __name__ == "__main__":
    print("check 1: ", cosine_similarity([1, 1], [2,2]))
    print("check 2: ", cosine_similarity([1,1], [1,0]))

    code_a = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""
    code_b = """
def sort_numbers(numbers):
    return sorted(numbers)
"""
    code_c = """
def read_file(path):
    with open(path) as f:
        return f.read()
"""

    vec_a, vec_b, vec_c = embed([code_a, code_b, code_c])

    print("向量维度: ", len(vec_a))
    print("A-B 两种排序", cosine_similarity(vec_a, vec_b))
    print("A-C（排序 vs 读文件）:", cosine_similarity(vec_a, vec_c))
    print("B-C（排序 vs 读文件）:", cosine_similarity(vec_b, vec_c))

