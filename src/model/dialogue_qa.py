import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from collections import defaultdict
from os import listdir
from os.path import isfile, join

# Hyperparameters
max_per_slot = 2 # Max number of candidate responses to extract (per iteration)
max_span_length = 20 # Max length of one answer span
min_score = 0 # Min score for an answer span

class QA_Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    
    def answer(self, text, questions):
        answerss = []
        for question in questions:
            inputs = self.tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]

            text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            answer_start_scores, answer_end_scores = self.model(**inputs, return_dict=False)

            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
            score = torch.max(answer_start_scores) + torch.max(answer_end_scores)
            
            if answer_start < 1: 
                answer = "[None]" # cannot start with CLS
            elif answer_end - answer_start + 1 > max_span_length:
                answer = "[None]" # cannot be longer than hyperparam
            elif score < min_score:
                answer = "[None]" # cannot be < hyperparam
            else:
                answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
#                 print(f"Question: {question}")
#                 print(f"Answer: {answer}")
#                 print(f"Score: {score}") 
                
            answerss.append(answer)
        return answerss

class DialogueStateTracking:
    def __init__(self):
        self.qa = QA_Model()
        self.slot_temp = [] # slot template
        self.slot_questions = [] # questions corresponding to slot template
        
    # dialogue: list of strings (each utterance in a new string)
    # returns: answers: list of set of tuples. for each utterance, the slots predicted for the dialogue history
    def predict_slots(self, dialogue):
        answers = [] # length of dialogue
        for i in range(len(dialogue)):
            section = " ".join(dialogue[:i+1])
            answers_i = self.qa.answer(section, self.slot_questions)
            answers_i_dict = {}
            for j in range(len(answers_i)):
                # TODO: filter based on score, so you don't fill every slot every turn
                answers_i_dict[self.slot_temp[j]] = [answers_i[j]]
            answers.append(answers_i_dict)
        return answers

