import stanza
import json
zh_nlp = stanza.Pipeline('zh', use_gpu=False)

# text = "华兰生物(002007.SZ)5月26日在投资者互动平台表示，疫苗公司分拆上市已于2020年12月获深圳证券交易所受理，目前正处于反馈回复阶段。"
# doc = zh_nlp(text)
# print(doc.sentences)

for filename in ['test', 'train']:
    file_path = "/Users/zhangqi/Documents/event-extractor/EMNLP2018-JMEE-master/{}.json".format(filename)

    ans_data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for tmp in data[:]:
        # print(tmp)
        text = tmp['sentence']
        # print(text)
        doc = zh_nlp(text)
        ans_words = []
        ans_colcc = []
        ans_golden_entity_mentions = []
        ans_conll_head = []
        ans_pos_tags = []
        ans_lemma = []
        start_id = 0
        character_len = 0
        ans_event = []
        for event in tmp["events"]:
            tmp_ans = {
                "arguments": [
                ],
                "trigger": {
                    "start": None,
                    "end": None,
                    "start_character": event["trigger"]['offset'],
                    "end_character": event["trigger"]['offset'] + event["trigger"]['length'],
                    "text": "{}".format(event["trigger"]['text'])
                },
                "event_type": "{}".format(event['polarity'])
            }
            for role in event['arguments']:
                tmp_ans['arguments'].append({
                    "start": None,
                    "start_character": role['offset'],
                    "role": "{}".format(role['role']),
                    "end": None,
                    "end_character": role['offset']+role["length"],
                    "text": "{}".format(role["text"]),
                })
            ans_event.append(tmp_ans)
        for i in range(len(doc.sentences)):
            sent = doc.sentences[i]
            # print(sent)
            entity_start = None
            entity_text = None
            for j in range(len(sent.tokens)):
                token = sent.tokens[j]
                word = sent.words[j]
                # print(token.text, word.text, token.ner)
                assert token.text == word.text
                if word.id == 0 or word.text == "root" or word.text is None:
                    start_id -= 1
                    continue
                # print(word.id+start_id, token.text, token.ner, word.head+start_id, word.lemma, word.deprel, word.pos)
                ans_words.append(word.text)
                ans_colcc.append("{}/dep={}/gov={}".format(word.deprel, word.id+start_id, word.head+start_id))
                ans_conll_head.append(word.head+start_id)
                ans_pos_tags.append(word.pos)
                ans_lemma.append(word.lemma)
                if token.ner == "O":
                    entity_text = ""
                    entity_start = None
                elif token.ner.split("-")[0] == "S":
                    ans_golden_entity_mentions.append({
                        "end": word.id+start_id+1,
                        "text": "{}".format(word.text),
                        "entity-type": "{}".format(token.ner.split("-")[1]),
                        "start": word.id+start_id
                    })
                elif token.ner.split("-")[0] == "E":
                    if entity_text is None:
                        pass
                    else:
                        entity_text += word.text
                        ans_golden_entity_mentions.append({
                            "end": word.id+start_id+1,
                            "text": "{}".format(entity_text),
                            "entity-type": "{}".format(token.ner.split("-")[1]),
                            "start": entity_start
                        })
                        entity_start = None
                        entity_text = None
                elif token.ner.split("-")[0] == "B":
                    entity_start = word.id+start_id+1
                    entity_text = word.text
                elif token.ner.split("-")[0] == "I":
                    if entity_text is None:
                        pass
                    else:
                        entity_text += word.text

                for event in ans_event:
                    if event['trigger']['start'] is None and event['trigger']['start_character']+1 <= character_len+len(word.text):
                        event['trigger']['start'] = word.id+start_id -1

                    if event['trigger']['end'] is None and event['trigger']['end_character'] <= character_len+len(word.text):
                        event['trigger']['end'] = word.id+start_id
                    for role in event['arguments']:
                        # print(role, role['start'])
                        if role['start'] is None and role['start_character']+1 <= character_len + len(word.text):
                            # print(role['start_character'], character_len, len(word.text), word.id, start_id)
                            role['start'] = word.id - 1 + start_id
                        if role['end'] is None and role['end_character'] <= character_len + len(word.text):

                            role['end'] = word.id + start_id
                character_len += len(word.text)

            start_id += len(sent.tokens)

        for event in ans_event:
            # print(ans_words[event['trigger']["start"]: event['trigger']["end"]], event['trigger']["start"], event['trigger']["end"], event['trigger']["text"])
            event['trigger']['text_'] = "".join(ans_words[event['trigger']["start"]: event['trigger']["end"]])
            for role in event['arguments']:
                # print(ans_words[role["start"]: role["end"]], role["start"],
                #       role["end"], role["text"])
                role['text_'] = "".join(ans_words[role["start"]: role["end"]])

        ans_tmp = {}
        ans_tmp['words'] = ans_words
        ans_tmp["stanford-colcc"] = ans_colcc
        ans_tmp["golden-entity-mentions"] = ans_golden_entity_mentions
        ans_tmp["conll-head"] = ans_conll_head
        ans_tmp["lemma"] = ans_lemma
        ans_tmp["pos-tags"] = ans_pos_tags
        ans_tmp["golden-event-mentions"] = ans_event
        ans_data.append(ans_tmp)
        # print(ans_tmp)

    with open("{}.json".format(filename), 'w', encoding='utf-8') as f:
        json.dump(ans_data, f, ensure_ascii=False)