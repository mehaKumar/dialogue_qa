import numpy as np
import re
import collections

# modified from https://github.com/jasonwu0731/trade-dst/, https://rajpurkar.github.io/SQuAD-explorer/

# gets metrics for a all dialogue conversations
# all_diag_pred: dictionary of conversation_name -> diag_pred
# all_diag_gold: dictionary of conversation_name -> diag_gold
# diag_pred: list of predicted slots per dialogue turn (list of sets of tuples)
# diag_gold: list of gold slots per dialogue turn (list of sets of tuples)
# slot_temp: set of slots we are concerned about (template)
def evaluate_metrics(diag_pred, diag_gold, slot_temp):
    total, turn_acc, f1_pred = 0, 0, 0
    for t in range(len(diag_gold)):
        curr_pred = diag_pred[t]
        curr_gold = diag_gold[t] 
        total += 1

        acc = compute_acc(curr_gold, curr_pred, slot_temp)
        f1 = compute_f1(curr_gold, curr_pred, slot_temp)
        turn_acc += acc
        f1_pred += f1

    turn_acc_score = turn_acc / float(total) if total!=0 else 0
    F1_score = f1_pred / float(total) if total!=0 else 0
    return F1_score, turn_acc_score

# gold, pred: dictionaries of slot -> list of values
# slot_temp: set of slots we are concerned about (template)
# returns: (# slots which have any correct prediction) / (# slots)
def compute_acc(gold, pred, slot_temp):
    pred_correct = 0
    # total up number of slots where any gold value was predicted
    for slot in slot_temp:
        for gold_val in gold[slot]:
            if gold_val in pred[slot]:
                pred_correct += 1
                break
    
    ACC_TOTAL = len(slot_temp)
    ACC = pred_correct
    ACC = ACC / float(ACC_TOTAL)
    return ACC

# gold, pred: dictionaries of slot -> list of values
def compute_prf(gold, pred, slot_temp):
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        for slot in slot_temp:
            for gold_val in gold[slot]:
                if gold_val in pred[slot]:
                    TP += 1
                else:
                    FN += 1
            for pred_val in pred[slot]:
                if pred_val not in gold[slot]:
                    FP += 1
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
#     else:
#         if len(pred)==0:
#             precision, recall, F1, count = 1, 1, 1, 1
#         else:
#             precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision

def compute_bleu(gold, pred):
    from sacrebleu.metrics import BLEU
    def get_bleu(gold, pred):
        # return BLEU, bleu-1, -2, -3, -4
        x = bleu.corpus_score([pred], [[gold]])
        b1, b2, b3, b4 = [float(s) for s in x._verbose.split()[0].split('/')]
        return {'bleu': [x.score], 'bleu-1': [b1], 'bleu-2': [b2], 'bleu-3': [b3], 'bleu-4': [b4]}
    
    bleu = BLEU()
    miss_gold = 0
#     miss_slot = []
    scorer = {'bleu': [], 'bleu-1': [], 'bleu-2': [], 'bleu-3': [], 'bleu-4': []}
    for g in gold:
        if g not in pred:
            miss_gold += 1
#             miss_slot.append(g[0])
#     wrong_pred = 0
    for s, v in pred:
        d_gold = dict(gold)
        if s in d_gold:
            bleus = get_bleu(v, d_gold[s])
            scorer = {key:scorer.get(key,[])+bleus.get(key,[]) 
                      for key in set(list(scorer.keys())+list(bleus.keys()))}
    for i in range(miss_gold):
        scorer = {k: scorer[k] + [0.] for k in scorer}
    return {k: np.mean(scorer[k]) for k in scorer} #, scorer

# Word-based F1, as used for span prediction
# a_fold, a_pred: strings (spans of text)
def compute_f1_span(a_gold, a_pred):
    gold_toks = a_gold.split() #TODO: more appropriate tokenization?
    pred_toks = a_pred.split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# word-based F1: average of slots' F1 scores, where a slot's score is the max F1 of all gold/pred pairs
def compute_f1(gold, pred, slot_temp):
    f1s = []
    for slot in slot_temp:
        # if gold answer was no answer, f1 is 1 if pred is same or 0 if not
        if len(gold[slot]) == 0:
            f1s.append(int(len(gold[slot]) == len(pred[slot])))
        else:
            f1_max = 0
            for gold_val in gold[slot]:
                for pred_val in pred[slot]:
                    f1 = compute_f1_span(gold_val, pred_val)
                    f1_max = max(f1_max, f1)
            f1s.append(f1_max)
    return np.mean(f1s)

def annotation_to_gold(annot_lines, diag_lines, slot_temp):
    # turn annotation file into list of dict
    slot_2_tag = {}
    for s in slot_temp:
        slot_2_tag[s] = []

    #make dict of tags to location and string
    annot_dict = {}
    for line in annot_lines:
        toks = re.split(r'\t+', line)
        if toks[0][0] != "E":
            annot_dict[toks[0]] = (toks[1], toks[2].strip())
            
    # print(annot_dict)

    #make dict of slot needed to tag(s)
    for line in annot_lines:
        if line[:2] == "E1":
            #found the theft event
            toks = line[2:].split()
            for tok in toks:
                tag = tok.split(":")
                for s in slot_temp:
                    if tag[0].startswith(s):
                        slot_2_tag[s] += [tag[1]]
                        break
              

    # print(slot_2_tag)

    #make list of dicts (entry i is slot values at utterance i)
    slots_gold = []

    len_total = 0
    for line in diag_lines:
        len_total += len(line)
        slots_gold_i = {}
        for s in slot_temp:
            slots_gold_i[s] = []
        # for each slot, add any gold annotated strings which are in the dialogue history
        for slot in slot_temp:
            tags = slot_2_tag[slot]
            for tag in tags:
                loc = annot_dict[tag][0]
                toks = loc.split()
                if int(toks[1]) <= len_total:
                    slots_gold_i[slot].append(annot_dict[tag][1])
        slots_gold.append(slots_gold_i)
        
    # print(slots_gold)
    return slots_gold


# testing
# slot_temp = ["Target_Object-ARG", "Place-Arg", "Start_Time-Arg", "End_Time-Arg"]
# f = open('data/val_gold/event_1206297.ann')
# annot_lines = f.readlines()
# f.close()
# f = open('data/val/event_1206297.txt')
# diag_lines = f.readlines()
# f.close()
# slots_gold = annotation_to_gold(annot_lines, diag_lines, slot_temp)
# print(slots_gold)
# gold = {"Target_Object-ARG":["bike lights (both front and rear)"], "Place-Arg": ["in front of [ORG]"],
#         "Start_Time-Arg": ["This evening ([DATE]), between #:##"], "End_Time-Arg": ["#:##"]}

# pred = {"Place-Arg": ["in front of [ORG]"], "Target_Object-ARG":["bike lights"], 
#         "End_Time-Arg": ["This evening"], "Start_Time-Arg":[]}

# print(compute_acc(gold, pred, slot_temp))
# print(compute_prf(gold, pred, slot_temp))
# print(compute_f1(gold, pred, slot_temp))
# # print(compute_bleu(gold, pred))