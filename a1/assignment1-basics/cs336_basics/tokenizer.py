import regex as re
import os
import time
import heapq
import json

from tqdm import tqdm
from collections import defaultdict
from typing import Iterable, Iterator
# GPT-2 分词正则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class DLLNode:
    __slots__ = ['v', 'prev', 'next']
    def __init__(self, v):
        self.v = v
        self.prev: 'DLLNode | None' = None
        self.next: 'DLLNode | None' = None

class HeapItem:
    __slots__ = ['count', 'token_id1', 'token_id2', 'vocab_ref', 'bytes1', 'bytes2']
    def __init__(self, count, token_id1, token_id2, vocab_ref):
        self.count = count
        self.token_id1 = token_id1
        self.token_id2 = token_id2
        self.vocab_ref = vocab_ref
        self.bytes1 = vocab_ref[token_id1]
        self.bytes2 = vocab_ref[token_id2]

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        if self.bytes1 != other.bytes1:
            return self.bytes1 > other.bytes1
        return self.bytes2 > other.bytes2
    
    def __eq__(self, other):
        return (self.count == other.count and self.bytes1 == other.bytes1 and self.bytes2 == other.bytes2)
     
    def get_pair(self):
        return (self.token_id1, self.token_id2)
    
def bytes_to_unicode():
    """
    返回 GPT-2 使用的字节到 Unicode 字符的映射字典。
    这是 'GPT-2 规则' 的核心：避免不可打印字符，把 byte 映射为可见字符。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + 
        list(range(ord("¡"), ord("¬") + 1)) + 
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """bpe训练"""
    # 读取文件
    print(f"正在读取文件: {input_path}...")
    with open(input_path, "rb") as f:
        text_bytes = f.read()
    text = text_bytes.decode("utf-8", errors="replace")

    # 预分词
    print("正在进行正则预分词 ...")
    if special_tokens:
        escaped_tokens = [re.escape(t) for t in special_tokens]
        special_pattern = "(" + "|".join(escaped_tokens) + ")"
        parts = re.split(special_pattern, text)
    else:
        parts = [text]
    
    gpt2_pat = re.compile(PAT)
    words_list = []
    for part in tqdm(parts, desc="处理文本块"):
        if part in special_tokens:
            continue
        found = gpt2_pat.findall(part)
        for word in tqdm(found, desc="编码UTF-8字节", leave=False):
            words_list.append([b for b in word.encode("utf-8")])

    # 构建双向链表
    print("正在构建双向链表...")
    head_nodes = []
    stats = defaultdict(int)
    indices = defaultdict(list)

    for word in tqdm(words_list, desc="构建链表"):
        if not word:
            continue

        head = DLLNode(word[0])
        head_nodes.append(head)
        prev = head
        for i in range(1, len(word)):
            curr = DLLNode(word[i])
            prev.next = curr
            curr.prev = prev

            pair = (prev.v, curr.v)
            stats[pair] += 1
            indices[pair].append(prev)
            prev = curr

    # 初始化堆
    vocab = {i : bytes([i]) for i in range(256)}
    pq = []
    for pair, count in tqdm(stats.items(), desc="初始化堆"):
        Item = HeapItem(count, pair[0], pair[1], vocab)
        heapq.heappush(pq, Item)

    # 合并
    print("开始执行 BPE 合并...")
    merges = []
    num_merges = vocab_size - 256 - len(special_tokens)

    for i in tqdm(range(num_merges), desc="BPE合并中"):
        # 找到要合并的对
        best_pair = None
        while pq:
            item = heapq.heappop(pq)
            cur_pair = item.get_pair()
            cur_count = item.count
            
            if cur_count != stats.get(cur_pair, 0):
                continue

            best_pair = cur_pair
            break
        
        if best_pair is None:
            break

        # 记录合并
        new_id = 256 + i
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        new_token_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[new_id] = new_token_bytes

        # 合并
        nodes = indices[best_pair]
        del stats[best_pair]
        del indices[best_pair]

        for node in nodes:
            # 有效性检查
            if node.v != best_pair[0] or node.next is None or node.next.v != best_pair[1]:
                continue
            
            # 更新
            prev_node = node.prev
            next_node = node.next
            next_next_node = next_node.next
            # 更新左邻居
            if prev_node:
                old_pair = (prev_node.v, node.v)
                if old_pair != best_pair:
                    stats[old_pair] -= 1
                    if stats[old_pair] == 0:
                        del stats[old_pair]
                    else:
                        heapq.heappush(pq, HeapItem(stats[old_pair], prev_node.v, node.v, vocab))

                new_pair = (prev_node.v, new_id)
                stats[new_pair] += 1
                indices[new_pair].append(prev_node)
                heapq.heappush(pq, HeapItem(stats[new_pair], prev_node.v, new_id, vocab))

            # 更新右邻居
            if next_next_node:
                old_pair = (next_node.v, next_next_node.v)
                if old_pair != best_pair:
                    stats[old_pair] -= 1
                    if stats[old_pair] == 0:
                        del stats[old_pair]
                    else:
                        heapq.heappush(pq, HeapItem(stats[old_pair], next_node.v, next_next_node.v, vocab))
                
                new_pair = (new_id, next_next_node.v)
                stats[new_pair] += 1
                indices[new_pair].append(node)
                heapq.heappush(pq, HeapItem(stats[new_pair], new_id, next_next_node.v, vocab))

            # 物理合并
            node.v = new_id
            node.next = next_next_node
            if next_next_node:
                next_next_node.prev = node

            next_node.v = -1
    
    next_id = 256 + len(merges)

    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1
    return vocab, merges

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None):
        """
        初始化分词器
        :param vocab: token id 到字节串的映射
        :param merges: 合并规则列表(字节串1, 字节串2)
        :paarm special_tokens: 特殊token列表
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        # 构建反向词表(bytes->id)
        self.encoder = {v: k for k, v in self.vocab.items()}
        # 构建合并规则映射优先级(pair->rank)
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.pat = re.compile(PAT)
        # 编译特殊token正则，用于split
        if special_tokens:
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(t) for t in sorted_special_tokens]
            self.special_pat = re.compile(r"(" + "|".join(escaped_tokens) + r")")
        else:
            self.special_pat = None
        # 检查特殊token是否已经在词表中
        cur_max_id = max(self.vocab.keys()) if self.vocab else 255
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.encoder:
                cur_max_id += 1
                self.vocab[cur_max_id] = st_bytes
                self.encoder[st_bytes] = cur_max_id

    def decode(self, ids: list[int]) -> str:
        """
        将id序列解码为文本
        """
        byte_parts = []
        for id in ids:
            if id in self.vocab:
                byte_parts.append(self.vocab[id])
            else:
                pass
        
        full_bytes = b"".join(byte_parts)
        text = full_bytes.decode("utf-8", errors="replace")
        return text
    
    def _iterate_nodes(self, node):
        while node:
            yield node
            node = node.next

    def _bpe(self, token_bytes: bytes) -> list[int]:
        """
        将单个token的字节序列转化为id序列
        """
        tokens = [bytes([b]) for b in token_bytes]
        prev = None
        head = None
        # 建立双向链表
        for token in tokens:
            curr = DLLNode(token)
            curr.prev = prev
            if prev is not None:
                prev.next = curr
            else:
                head = curr
            prev = curr

        # 找到当前要合并的对
        while True:
            curr = head
            best_pair = None
            lowest_rank = float("inf")
            while curr:
                if curr.next is None:
                    break

                curr_pair = (curr.v, curr.next.v)
                curr_rank = self.bpe_ranks.get(curr_pair, float("inf"))
                if curr_rank < lowest_rank:
                    best_pair = curr_pair
                    lowest_rank = curr_rank
                
                curr = curr.next
            
            # 没有要合并的对
            if lowest_rank == float("inf"):
                break
            # 合并
            curr = head
            while curr:
                if curr.next is None:
                    break

                curr_pair = (curr.v, curr.next.v)
                next = curr.next
                if curr_pair == best_pair:
                    curr.v = curr.v + next.v
                    curr.next = next.next
                    if next.next is not None:
                        next.next.prev = curr

                curr = curr.next

        ids = []
        curr = head
        while curr:
            ids.append(self.encoder[curr.v])
            curr = curr.next
        return ids
 
    def encode(self, text: str) -> list[int]:
        """
        将文本转化为id序列
        """
        ids = []

        # 将文本按照特殊token切分
        if self.special_pat:
            parts = self.special_pat.split(text)
        else:
            parts = [text]
        for part in tqdm(parts, desc="Tokenizing"):
            part_bytes = part.encode("utf-8")
            if part_bytes in self.encoder and part in self.special_tokens: # 特殊token
                ids.append(self.encoder[part_bytes])
            else: # 普通文本
                pre_tokekns = self.pat.findall(part)
                for token in pre_tokekns:
                    token_bytes = token.encode("utf-8")
                    token_ids = self._bpe(token_bytes)
                    ids.extend(token_ids)
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        处理大型文本流,惰性翻译ID
        """
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        从符合 GPT-2 格式的文件中加载 Tokenizer。
        """

         # 获取映射表并反转
        byte_encoder = bytes_to_unicode()
        byte_decoder = {v : k for k, v in byte_encoder.items()}
        
        # 加载词表 (vocab.json)
        # GPT-2 格式: {"TokenString": ID}
        # 目标格式: {ID: b"TokenBytes"}
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_reversed = json.load(f)

        vocab = {}
        for token_str, token_id in vocab_reversed.items():
            token_bytes = bytes([byte_decoder[c] for c in token_str])
            vocab[token_id] = token_bytes
        
        # 加载合并规则
        # GPT-2 格式: 每行两个 token，空格隔开 (例如: "Ġ t")
        # 第一行通常是注释 "#version: 0.2"，需要跳过
        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            lines = f.read().splitlines()
        
        start_idx = 0
        if lines and lines[0].startswith("#version"):
            start_idx = 1
        
        for line in lines[start_idx:]:
            if not line.strip():
                continue

            parts = line.split(" ")
            if len(parts) != 2:
                continue

            token_bytes1 = bytes([byte_decoder[c] for c in parts[0]])
            token_bytes2 = bytes([byte_decoder[c] for c in parts[1]])
            merges.append((token_bytes1, token_bytes2))
        
        return cls(vocab, merges, special_tokens)
