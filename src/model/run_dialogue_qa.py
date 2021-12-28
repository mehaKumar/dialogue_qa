from dialogue_qa import DialogueStateTracking
import evaluation
from os import listdir
from os.path import isfile, join
from numpy import mean

# script to run slot filling on ../data/ output answers and eval
dst = DialogueStateTracking()
dst.slot_temp = ["Target_Object-ARG", "Place-Arg", "Start_Time-Arg", "End_Time-Arg"]
dst.slot_questions = ["What was stolen?" ,
                     "Where did the theft take place?",
                     "When was the item last seen?",
                     "When was the item stolen?"]

val_data_path = "./data/val/"
val_gold_data_path = "./data/val_gold/"
fnames = [f for f in listdir(val_data_path) if isfile(join(val_data_path, f))]
accs = []
f1s = []
for fname in fnames:
    f = open(val_data_path + fname)
    dialogue = f.readlines()
    slots_pred = dst.predict_slots(dialogue)
    f.close()
    ann_name = fname[:-4] + ".ann"
    f = open(val_gold_data_path + ann_name)
    ann_lines = f.readlines()
    f.close()
    slots_gold = evaluation.annotation_to_gold(ann_lines, dialogue, dst.slot_temp)
    acc, f1 = evaluation.evaluate_metrics(slots_gold, slots_pred, dst.slot_temp)
    accs.append(acc)
    f1s.append(f1)

print(f"Number of dialogues: {len(fnames)}")
print(f"Average accuracy: {mean(accs):0.2f}")
print(f"Average F1: {mean(f1s):0.2f}")





