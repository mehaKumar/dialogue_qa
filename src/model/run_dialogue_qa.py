from dialogue_qa import DialogueStateTracking
import evaluation
from os import listdir
from os.path import isfile, join
from numpy import mean

from tqdm import tqdm

# script to run slot filling on ../data/ output answers and eval
model = "deepset/roberta-base-squad2"
dst = DialogueStateTracking(model)
dst.slot_temp = ["Target_Object-ARG", "Place-Arg", "Start_Time-Arg", "End_Time-Arg"]
dst.slot_questions = ["What object was stolen?" ,
                     "Where did the theft take place?",
                     "When did you last see the stolen object?",
                     "When did you notice the object was missing?"]

val_data_path = "./data/val/"
val_gold_data_path = "./data/val_gold/"
val_gold_dict_path = "./data/val_gold_dict/"
output_path = "./outputs/val_pred_dict/"
fnames = [f for f in listdir(val_data_path) if isfile(join(val_data_path, f))]
accs = []
f1s = []
for fname in tqdm(fnames):
    f = open(val_data_path + fname)
    dialogue = f.readlines()
    slots_pred = dst.predict_slots(dialogue)
    with open(output_path + fname, 'w') as f_p:
        for s_p in slots_pred:
            f_p.write('%s\n' % s_p)

    f.close()
    ann_name = fname[:-4] + ".ann"
    f = open(val_gold_data_path + ann_name)
    ann_lines = f.readlines()
    f.close()
    slots_gold = evaluation.annotation_to_gold(ann_lines, dialogue, dst.slot_temp)
    
    with open(val_gold_dict_path + fname, 'w') as f_g:
        for s_g in slots_gold:
            f_g.write('%s\n' % s_g)

    f1, acc = evaluation.evaluate_metrics(slots_gold, slots_pred, dst.slot_temp)
    accs.append(acc)
    f1s.append(f1)

print(f"Number of dialogues: {len(fnames)}")
print(f"Average accuracy: {mean(accs):0.2f}")
print(f"Average F1: {mean(f1s):0.2f}")

f = open("outputs/ablations.txt", 'a')
f.write(f"Model: {model}\n")
f.write("Questions: \n")
f.write(str(dst.slot_questions) + "\n")
f.write("Slots: \n")
f.write(str(dst.slot_temp) + "\n")
f.write(f"Number of dialogues: {len(fnames)}\n")
f.write(f"Average accuracy: {mean(accs):0.2f}\n")
f.write(f"Average F1: {mean(f1s):0.2f}\n")
f.write("====================================\n")
f.close()







