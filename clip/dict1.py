import json

def convert_json_to_txt(json_file, txt_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(txt_file, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(entry['Entry'] + '\n')

# 使用示例
convert_json_to_txt('glosses.json', 'dict.txt')