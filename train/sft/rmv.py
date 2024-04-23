def append_lines_to_odd_lines(source_file, target_file):
    with open(source_file, 'r', encoding='utf-8') as source, open(target_file, 'r', encoding='utf-8') as target:
        source_lines = source.readlines()
        target_lines = target.readlines()

    with open(target_file, 'w', encoding='utf-8') as target:
        for i, line in enumerate(target_lines, start=1):
            target.write(line.rstrip())
            if i % 2 != 0:  # 如果是奇数行
                if source_lines:  # 如果源文件还有行
                    target.write("match:" + source_lines.pop(0).rstrip())  # 从源文件中弹出一行并写入目标文件
            target.write('\n')
# 使用示例
append_lines_to_odd_lines('match.txt', 'all_with_match.txt')