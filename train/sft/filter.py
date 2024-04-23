import re

# 定义一个函数,用于去除"*N*"和"-"字符
def remove_special_chars(text):
    filtered = re.sub(r'\*\d+\*|-', '', text)
    return filtered

# 定义一个函数,用于去除非中英文和#字符
def remove_non_chinese_english_hash(text):
    filtered = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5#]', '', text)
    return filtered
def remove_brackets_content(text):
    filtered = re.sub(r'\(.*?\)', '', text)
    return filtered
# 指定输入文件路径
input_file = 'all.txt'

# 指定输出文件路径
output_file = 'filter.txt'

# 打开输入文件和输出文件
with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    # 遍历输入文件的每一行
    for i, line in enumerate(f_in, start=1):
        # 如果是偶数行
        if i % 2 == 0:
            # 去除"*N*"和"-"字符
            line = remove_special_chars(line)
            line = remove_brackets_content(line)
            # 去除非中英文和#字符
            filtered_line = remove_non_chinese_english_hash(line)
            # 写入到输出文件
            f_out.write(filtered_line)
            f_out.write("\n")
        else:
            # 直接写入到输出文件
            f_out.write(line)