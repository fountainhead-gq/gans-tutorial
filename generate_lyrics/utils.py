import os, glob
import time
import jieba


def merge_txt(dir_path, txt_name):
    txt_file=open(txt_name, 'w', encoding='utf-8')  
    file_names = os.listdir(dir_path)
    for file_name in file_names:
        file_path = os.path.join(dir_path,file_name)
        for txt in open(file_path, encoding='utf-8'):
            txt_file.writelines(txt)
        txt_file.write('\n')
    txt_file.close()
	print('done.')
    
    
def preprocess(data):
    """
    对文本中的字符进行替换
    :param text: 
    :return: 
    """
    data = data.replace('《', '')
    data = data.replace('》', '')
    data = data.replace('【', '')
    data = data.replace('】', '')
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = data.replace('<', '')
    data = data.replace('>', '')
    data = data.replace(' ', ';')
    data = data.replace('\n', '')
    # data = data.replace('\n', '.')
    # words = jieba.lcut(data, cut_all=True) 
    words = jieba.lcut(data, cut_all=False) 
    return words    


def write_file(words, fname):
    with open(fname, 'a', encoding='utf-8') as f:
        for w in words:
            f.write(w + '\n')

            
def split_txt(file_txt, new_txt):
    with open(file_txt, encoding='utf-8') as f:
        split_text = f.read()
    proce_words = preprocess(split_text)
    write_file(proce_words, new_txt)
    print('done.')