import jieba

# 加载用户词典
#jieba.load_userdict('user_dict.txt')
def extract_odd_lines(file_path):
    odd_lines = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines, start=1):
            if i % 2 != 0:
                odd_lines.append(line.strip())
    return odd_lines

def precise_tokenize(text):
    # 精确模式分词
    tokens = jieba.cut(text, cut_all=False, HMM=False)
    return list(tokens)

# 示例用法
file_path = "../../data/20kdata/filtered_new_all.txt"
#sentences = extract_odd_lines(file_path)
with open('../../data/20kdata/yfs.txt') as f:
    sentences = f.readlines()
for text in sentences:
    tokens = precise_tokenize(text.strip())
    print("#".join(tokens))