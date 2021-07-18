import stanza
import json
zh_nlp = stanza.Pipeline('zh', use_gpu=False)

# text = "华兰生物(002007.SZ)5月26日在投资者互动平台表示，疫苗公司分拆上市已于2020年12月获深圳证券交易所受理，目前正处于反馈回复阶段。"
# doc = zh_nlp(text)
# print(doc.sentences)
def got_voc():
    word_set = set()
    for filename in ["train", 'dev', 'test']:
        file_path = "/Users/zhangqi/Documents/event-extractor/EMNLP2018-JMEE-master/{}.json".format(filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for tmp in data[:20]:
            # print(tmp)
            text = tmp['sentence']
            doc = zh_nlp(text)
            for i in range(len(doc.sentences)):
                sent = doc.sentences[i]
                for j in range(len(sent.tokens)):
                    word = sent.words[j]
                    word_set.add(word.text)

    with open("voc.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(list(word_set)))


def got_small_tengxun_voc():
    word_set = {}
    with open("voc.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            word_set[line.strip()] = 1

    ans_str = []
    with open("/Users/zhangqi/Downloads/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt", 'r', encoding='utf-8') as f:
        line = f.readline()
        flag = 0
        while len(line.strip())>0:
            if flag == 0:
                flag = 1
                continue
            tmp = line.strip().split(" ")
            if word_set.get(" ".join(tmp[:-200]), 0) == 1:
                ans_str.append(line.strip())
            line = f.readline()

    with open("voc_embeding.txt", "w", encoding='utf-8') as f:
        f.write("{} {}\n".format(len(ans_str), 200))
        f.write("\n".join(ans_str))


if __name__ == '__main__':
    # got_voc()
    got_small_tengxun_voc()