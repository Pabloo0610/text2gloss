import mysql.connector
import numpy as np
import string
import jieba
import jieba.posseg as pseg
import time
import pandas as pd
import re
import json

from tqdm import tqdm
import os

import hanlp
hanlp.pretrained.sts.ALL

class WordMatch:
    def __init__(self):
        
        self.m_data = self.GetData()
        
        cnPunctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。"
        allPunctuation = string.punctuation + cnPunctuation
        # 问号,短杠,波浪线,六个点和冒号,省略号有意义，忽略
        self.m_allPunctuation = (allPunctuation.replace("……", "").replace("～", "").replace("~", "").replace("-", "").replace(":", "")).replace("?","").replace("？","")
        
        self.m_wordSet = self.CreateSet(self.m_data)
        
        for word in self.m_wordSet:
            jieba.add_word(word, tag="ours")
            
        self.m_sts = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)
        self.m_segmentWords = []
        self.m_synonyms = {}
        self.m_supplementMatch = {}
        self.m_notMatchedWords = []
    
        self.m_sentence = ''
    

    def GetData(self):
        # 连接到 MySQL 数据库（自定义端口号）
        mydb = mysql.connector.connect(
            host="usl.dgene.tech",
            user="chengang",
            password="rhythmo123",
            database="shsy",
            port="33060",  # 自定义端口号，例如 33060
        )
        # 创建一个游标对象
        mycursor = mydb.cursor()
        # 查询数据库
        # mycursor.execute("SELECT * FROM vocabulary")
        mycursor.execute("SELECT id, Entry, Synonyms  from vocabulary v WHERE AuditResults = 1;")
        
        # 获取查询结果
        result = mycursor.fetchall()
        # 关闭数据库连接
        mydb.close()
        return result


    def CreateSet(self, result):
        
        wordSet = set()
        for i, row in enumerate(result):
            # row: (id, Entry, Synonyms)
            synonyms = [i.translate(str.maketrans("", "", self.m_allPunctuation))for i in row[2].split("/")]  
 
            for synonym in synonyms:
                wordSet.add(synonym)

        print("Words data loaded! Number of words: ",len(wordSet))
        return wordSet


    def jiebaSeg(self):
        words = pseg.cut(self.m_sentence)
        
        preservedFlags = ['ours', 'n', 'v', 'a', 's', 't', 'nr', 'ns', 'nt', 'nz', 'vn', 'an', 'ad', 'c', 'df', 'd', 'f', 'i', 'j', 'l', 'nr', 'z', 'r']

        notMatchPreservedWords = [] # 没有匹配上，但是有意义的词，继续查找近义词或精细匹配
        
        for word, flag in words:
            if flag in preservedFlags:
                self.m_segmentWords.append(word)
                
                if not flag == 'ours':
                    notMatchPreservedWords.append({'flag': flag, 'word': word, 'synonym': None, 'thoroughMatch': None})
                    self.m_notMatchedWords.append(word)
            else:
                self.m_notMatchedWords.append(word)
                
        return notMatchPreservedWords
            
    def SupplementSeg(self, inputStr):
       
        result = []
        # 循环直至输入字符串为空
        while inputStr:
            matchedWords = []
            # 获取匹配的单词并按照长度降序排列
            for word in self.m_wordSet:
                if inputStr.startswith(word):
                    matchedWords.append(word)

            matchedWords = sorted(matchedWords,key=len,reverse=True)
            
            if matchedWords[0] == '':
                matchedWords = None 

            # 如果找到匹配的单词，将其添加到结果列表
            if matchedWords:
                matchedWord = matchedWords[0]
                result.append(matchedWord)
                inputStr = inputStr[len(matchedWord):]
            else:  
            # 如果未找到匹配的单词，则跳过输入字符串的第一个字符并继续
                inputStr = inputStr[1:]
    
        if len(result) == 0:
            return None
        return result

    def WordCheck(self, wordSet, inputStr):
        for word in wordSet:
            if inputStr.startswith(word):
                print(word)


    def HanLP_sts(self, compareWord, topN = 3, threshold = 0.9):
        
        wordList = list(self.m_wordSet)
        compareList = [(word, compareWord) for word in wordList]
        #print(compareWord)
        similarityList =  self.m_sts(compareList)
 
        sorted_indices = sorted(range(len(similarityList)), key=similarityList.__getitem__, reverse=True)
        result = []
        for i in sorted_indices[:topN]:
            result.append(wordList[i])
            
        if len(result) == 0:
            return None
        return result
    
    
    def PostProcess(self, notMatchPreservedWords):
        '''
        item in notMatchPreservedWords: {'flag': flag, 'word': word, 'synonym': None, 'thoroughMatch': None}
        查找近义词, 都来源于词库
        '''
        
        for word in notMatchPreservedWords:
            word['synonym'] = self.HanLP_sts(word['word'])
            
        
        for word in notMatchPreservedWords:
            word['thoroughMatch'] = self.SupplementSeg(word['word'])    
            
        for word in notMatchPreservedWords:
            if word['synonym']:
                self.m_synonyms[word['word']] = word['synonym']
                
            if word['thoroughMatch']:
                self.m_supplementMatch[word['word']] = word['thoroughMatch']
                
            
            
                
    def Match(self, sentence, method = 'jieba', showResult = False):
        self.m_sentence = sentence
        self.m_segmentWords = []
        self.m_synonyms = {}
        self.m_supplementMatch = {}
        self.m_notMatchedWords = []
        
        if method == 'jieba':
            preservedWords = self.jiebaSeg()
        
        self.PostProcess(preservedWords)

        if showResult:
            print("---------Sentence:", self.m_sentence, "---------")
            print("Segment Words:", self.m_segmentWords)
            print("Not matched words:", self.m_notMatchedWords)
            print("Find synonym result:", self.m_synonyms)
            print("Supplementary search result:", self.m_supplementMatch)
        
    def MatchSentences(self, sentences, method = 'jieba'):
        resultList = []
        for sentence in tqdm(sentences, desc="Processing sentences"):
            self.Match(sentence, method)
            resultList.append({'sentence': sentence,
                               'segmentWords': self.m_segmentWords, 
                               'notMatchedWords': self.m_notMatchedWords,
                               'synonyms': self.m_synonyms, 
                               'supplementMatch': self.m_supplementMatch, 
                               })
        
        return resultList
   
def GenHTML(resultList, outputPath):
    html_content = '<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <title>Document</title>\n    <style>.red-text { color: red; }</style>\n</head>\n<body>\n'
    for item in resultList:
        wordsHasSupplement = [f"{word}({supplement})" for word, supplement in item["supplementMatch"].items()]
        wordsHasSynonym = [f"{word}({synonym})" for word, synonym in item["synonyms"].items()]
        
        html_content += f"<p>Sentence: {item['sentence']}</p>\n"
        segmentWords_html = ' / '.join([f"<span class='red-text'>{word}</span>" if word in item['notMatchedWords'] 
                                        else word for word in item['segmentWords']])
        html_content += f"<p>Segmented Words: {segmentWords_html}</p>\n"
        html_content += f"<p>Synonyms: {', '.join(wordsHasSynonym)}</p>\n"
        html_content += f"<p>Supplementary Match: {', '.join(wordsHasSupplement)}</p>\n"
        html_content += '<hr>\n'

    html_content += '</body>\n</html>'

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    outputPath = os.path.join(outputPath, 'output.html')
    
    with open(outputPath, 'w', encoding='utf-8') as file:
        file.write(html_content)

    print("HTML file generated successfully.")
    
def construct(w, supplement):
    dp = [None] * (len(w) + 1)
    dp[0] = []
    for i in range(1, len(w) + 1):
        for word in supplement:
            if i >= len(word) and w[i - len(word):i] == word and dp[i - len(word)] is not None:
                dp[i] = dp[i - len(word)] + [word]
                break
    return dp[-1]
    
def GenTXT(resultList, outputPath):
    
    txt_content = ''
    for item in resultList:
        segmentWords = item["segmentWords"]
        hasSupplement = item["supplementMatch"].keys()
        hasSynonym = item["synonyms"].keys()
        matchWords = []
        for w in segmentWords:
            if w not in item["notMatchedWords"]:
                matchWords.append(w)
                continue
            if w in item["notMatchedWords"] and w in hasSupplement:
                supplement = item["supplementMatch"][w]
                supplement = construct(w, supplement) # 尝试能不能拼出原字符串
                if supplement:
                    matchWords += supplement
                    continue
                
            if w in item["notMatchedWords"] and w in hasSynonym:
                synonym = item["synonyms"][w]
                #matchWords.append(f"{w}({synonym})")
                continue 
            else:
                raise ValueError(f"Word '{w}' doesn't have supplement or synonym.")
        line = '/'.join(matchWords)
        txt_content += f"{line}\n"
        
    outputFile = os.path.join(outputPath, 'output.txt')
    with open(outputFile, 'w', encoding='utf-8') as file:
        file.write(txt_content)
        
def extract_odd_lines(file_path):
    odd_lines = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines, start=1):
            if i % 2 != 0:
                odd_lines.append(line.strip())
    return odd_lines
     
def main():
    wm = WordMatch()
    file_path = "../../data/20kdata/filtered_new_all.txt"
    sentences = extract_odd_lines(file_path)
    # wm.Match('为了不让聋人因为沟通的困难失去最好的治疗机会，我会在牙防所全程陪护，为聋人和医生建立沟通的桥梁！', showResult=True)
    # sentences = ["为了不让聋人因为沟通的困难失去最好的治疗机会，我会在牙防所全程陪护，为聋人和医生建立沟通的桥梁！",
    #              "你最近牙齿有什么不舒服的地方吗？",
    #              "请张开嘴，我们先做个全面检查。"]
    
    # read sentence from a excel file
    # data = pd.read_excel(r'test\牙防所分词比较_2.xlsx')
    # sentences = data.iloc[:, 0]
    print(sentences[1])
    segResult = wm.MatchSentences(sentences)
    # GenHTML(segResult, r'test')
    GenTXT(segResult, r'test')

if __name__ == "__main__":
    main()
