from sacrebleu.metrics import BLEU, CHRF, TER

with open('test_new2_gt.txt', 'r') as file:
    lines = file.readlines()

outputs = []
gts = []

for line in lines:
    gt = line.split('Gt:')[1].rstrip()
    #print(gt)
    assistant = line.split('Gt:')[0].split('Assistant: ')[1]
    #print(assistant)
    outputs.append(assistant)
    gts.append(gt)
#refs = [['会是#关于#打击#谣言#的#吗？','我#爱#你']]
#sys = ['求#关于#回复谣言#的消息！？','我#爱#你']
bleu = BLEU(tokenize = 'zh')

print(bleu.corpus_score(outputs,gts))