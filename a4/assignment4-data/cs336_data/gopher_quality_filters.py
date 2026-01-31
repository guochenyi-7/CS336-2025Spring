import nltk

from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def gopher_quality_filter(text: str):
    words = word_tokenize(text)

    if not words:
        return True
    text_length = len(words)

    # 长度过滤
    if text_length < 50 or text_length > 100000:
        return False
    
    # 平均单词长度过滤
    total_length = sum(len(w) for w in words)
    avg_length = total_length / text_length if text_length > 0 else 0

    if avg_length < 3 or avg_length > 10:
        return False
    
    # 行末省略号过滤
    lines = text.splitlines()
    
    count = 0
    for line in lines:
        if  line.strip().endswith("..."):
            count += 1

    if count / len(lines) > 0.3:
        return False

    # 字母过滤
    words_with_alpha = [w for w in words if any(c.isalpha() for c in w)]
    ratio = len(words_with_alpha) / text_length
    if ratio < 0.8:
        return False

    return True
