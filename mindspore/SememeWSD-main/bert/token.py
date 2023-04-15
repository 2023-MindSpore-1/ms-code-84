import collections
from mindspore import Tensor
def convert_to_unicode(text):
    """
    Convert text into unicode type.
    Args:
        text: input str.

    Returns:
        input str in unicode.
    """
    ret = text
    if isinstance(text, str):
        ret = text
    elif isinstance(text, bytes):
        ret = text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    return ret

#通过转换文档得到token索引字典
def vocab_to_dict_key_token(vocab_file):
    """Loads a vocab file into a dict, key is token."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r",encoding='utf-8') as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

#对文本的添加处理
def convert_token(sentence):
    id_list=[]
    id_list.append('[CLS]')
    for i in range(len(sentence)):
        id_list.append(sentence[i])
    id_list.append('[SEP]')
    return id_list

# 按Bert要求的格式转换,同时适应当前的tokenizer
#生成规范化句子
def sen_token(sen):
    #格式化
    bert_tokens = convert_token(sen)
    # 进行转换
    id_list = []
    tokenizer = vocab_to_dict_key_token('vocab.txt')
    for token in bert_tokens:
        try:
            id_list.append(tokenizer[token])
        except:
            # 针对未知的添加
            id_list.append(tokenizer['[UNK]'])
    input_ids = Tensor([id_list])
    return input_ids


