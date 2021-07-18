import json
import stanza

# stanza.download('zh')

if __name__ == "__main__":
    sentences = []
    events = []
    zh_nlp = stanza.Pipeline('zh')
    path = "/Users/zhangqi/Documents/event-extractor/EMNLP2018-JMEE-master/train.json"
    with open(path, "r", encoding="utf-8") as f:
        for js in json.load(f):
            sentences.append(js['sentence'])
            events.append(js['events'])

    index = 0
    result = []

    for text in sentences:
        start_id = 0
        doc = zh_nlp(text)

        colcc = []
        entity_mentions = []
        head = []
        lemma = []
        words = []
        tags = []
        event_mentions = []

        for sent in doc.sentences:
            print("Sentence：" + sent.text)  # 断句
            print("Tokenize：" + ' '.join(token.text for token in sent.tokens))  # 中文分词
            print("word id: " + ' '.join(f'{word.text}/{word.id + start_id}' for word in sent.words))
            print("UPOS: " + ' '.join(f'{word.text}/{word.upos}' for word in sent.words))  # 词性标注（UPOS）
            # print("XPOS: " + ' '.join(f'{word.text}/{word.xpos}' for word in sent.words))  # 词性标注（XPOS）
            # print("dependencies", sent.dependencies)
            print("lemma: " + ' '.join(f'{word.text}/{word.lemma}' for word in sent.words))
            print("head id: " + ' '.join(f'{word.text}/{word.head + start_id}' for word in sent.words))

            print("NER: " + ' '.join(f'{ent.text}/{ent.type}' for ent in sent.ents))  # 命名实体识别
            print("NER: " + ' '.join(f'{token.text}/{token.ner}' for token in sent.tokens))  # 命名实体识别

            colcc.append(''.join(f'/dep={word.id + start_id}/gov={word.head + start_id}' for word in sent.words))
            entity_mentions.extend([{
                                       "phrase-type": "",
                                       "text": ent.text,
                                       "entity-type": ent.type,
                                       "start": ent.start_char + start_id,
                                       "end": ent.end_char + start_id,
                                       "id": ""
                                   } for ent in sent.ents])

            head.extend([(word.head + start_id) for word in sent.words])
            lemma.extend([word.lemma for word in sent.words])
            words.extend([token.text for token in sent.tokens])
            tags.extend([word.upos for word in sent.words])

            start_id += len(sent.tokens)

        event_mentions = [{
            "event_type": event["polarity"],
            "trigger": {
                "start": event["trigger"]["offset"],
                "end": event["trigger"]["offset"] + event["trigger"]["length"],
                "text": event["trigger"]["text"],
            },
            "arguments": [{
                "role": arg["role"],
                "start": arg["offset"],
                "end": arg["offset"] + arg["length"],
                "entity-type": "",
                "text": arg["text"]
            } for arg in event["arguments"]]
        } for event in events[index]]

        result_json = {
            "penn-treebank": "",
            "stanford-colcc": colcc,
            "golden-entity-mentions": entity_mentions,
            "conll-head": head,
            "chunk": "",
            "lemma": lemma,
            "words": words,
            "pos-tags": tags,
            "golden-event-mentions": event_mentions
        }
        result.append(result_json)
        index = index + 1

    with open("ans_data.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
