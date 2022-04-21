from dialogue_qa import DialogueStateTracking
import evaluation
from os import listdir
from os.path import isfile, join
from numpy import mean

from tqdm import tqdm

# script to run slot filling on ../data/ output answers and eval
model = "deepset/roberta-base-squad2"
dst = DialogueStateTracking(model)

# HarassmentAbuse + TheftLostItem
domains = ["HarassmentAbuse", "TheftLostItem"]
slot_temp_dict = {
    "HarassmentAbuse": [
        "Target-ARG", "Place-Arg", "Attacker-Arg", "End_Time-Arg"
        ],
    "TheftLostItem": [
        "Target-ARG", "Place-Arg", "Attacker-Arg", "End_Time-Arg", "Target_Object-Arg", "Start_Time-Arg"
        ],
}
slot_questions_dict = {
    "HarassmentAbuse": [
        "Who is the victim?" ,
        "Where did the incident take place?",
        "Who is the attacker?",
        "When did this happen?",
        ],
    "TheftLostItem": [
        "Who is the victim?" ,
        "Where did the theft take place?",
        "Who is the attacker?",
        "When did you last see the stolen object?",
        "What object was stolen?",
        "When did you notice the object was missing?",
    ],
}

for d in domains:
    print("evaluating %s instances..."%d)
    dst.slot_temp = slot_temp_dict[d]
    dst.slot_questions = slot_questions_dict[d]
    val_data_path = "./data/subset_1/%s/"%d
    # val_gold_data_path = "./data/val_gold/"
    # val_gold_dict_path = "./data/val_gold_dict/"
    output_path = "./outputs/subset_1/%s/"%d
    fnames = [f for f in listdir(val_data_path) if isfile(join(val_data_path, f))]
    accs = []
    f1s = []
    for fname in tqdm(fnames):
        f = open(val_data_path + fname, encoding='utf8')
        dialogue = f.readlines()
        slots_pred = dst.predict_slots(dialogue)
        with open(output_path + fname, 'w') as f_p:
            for s_p in slots_pred:
                f_p.write('%s\n' % s_p)

        f.close()
        ann_name = fname[:-4] + ".ann"
    








