from dialogue_qa import DialogueStateTracking
import evaluation
from os import listdir
from os.path import isfile, join
from numpy import mean

from tqdm import tqdm

# script to run slot filling on ../data/ output answers and eval
model = "deepset/roberta-base-squad2"
dst = DialogueStateTracking(model)


# HarassmentAbuse
print("evaluating HarassmentAbuse instances...")
dst.slot_temp = ["Target-ARG", "Place-Arg", "Attacker-Arg", "End_Time-Arg"]
dst.slot_questions = ["Who is the victim?" ,
                     "Where did the incident take place?",
                     "Who is the attacker?",
                     "When did this happen?"]
val_data_path = "./data/val/"
# val_gold_data_path = "./data/val_gold/"
# val_gold_dict_path = "./data/val_gold_dict/"
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

    








