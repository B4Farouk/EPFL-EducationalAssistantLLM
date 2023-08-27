# author : Henrique Da Silva Gameiro, Farouk Boukil
# date   : 27-04-2023
# utility: This script is used to interact with the chatGPT.

import sys

import numpy.random as rand

import json
import jsonlines

import gpt_wrapper
from gpt_wrapper.chat import Chat

from tqdm import tqdm

import random
import roman

############################################
# GLOBALS
############################################

AFFIRMATION_TEXT = set(["", "y", "Y", "YES", "yes", "oui", "OUI", "O", "o", "ok", "OK"])

MODEL_ARGS = {
    "temperature": 0.7, # default is 0.7
    "max_tokens": 400, # default is 100
    "top_p": 0.9, # default is 0.9
    "presence_penalty": 0.1, # default is 0.1
    "frequency_penalty": 0.1 # default is 0.1   
}

############################################
# AUXILARY FUNCTIONS
############################################

### FILE READING FUNCTIONS

def read_args(argsfilename):
    with open(argsfilename, 'r') as file:
        lines = file.readlines()
    
    args = {}
    for line in lines:
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        parsed = line.strip().split('=')
        if len(parsed) > 1:
            arg, v = parsed
            args[arg] = v
        else:
            args[parsed[0]] = True
    
    return args
def read_api_key(args):
    # get the API key from the command line arguments if given
    key = args.get("--api-key", None)
    if key is not None:
        return key
    # otherwise, it should be in a file
    filename = args["--api-key-file"]
    with open(filename, 'r') as file:
        key = file.readline().strip()
    return key

def read_questions_data(args):
    a_filename = args.get("--q-file", None)
    
    if a_filename is None:
        raise ValueError("You should provide a questions file using --q-file=<filename>.")
    
    with open(a_filename, 'r') as file:
        questions_data = json.load(file)
    
      
    # separate the answerable questions and the non-answerable questions
    a_questions_data = []
    for question_data in questions_data:
        a_questions_data.append(question_data)
        
    return a_questions_data

"""
def read_predetermined_instructions(args):
    filename = args.get("--i-file", None)
    
    if filename is None:
        print("No predetermined instructions.")
        return None
    
    with open(filename, 'r') as file:        
        # read the instruction set of the each type of questions
        instructions_by_qtype = {}
        qtype_tag = None
        for line in file.readlines():
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            if line.startswith('~'):
                qtype_tag = line.strip().strip('~')
                instructions_by_qtype[qtype_tag] = []
            else:
                assert qtype_tag is not None, "No question type tag given before the list of instructions."
                instructions_by_qtype[qtype_tag].append(line.strip())
        return instructions_by_qtype
"""

def define_instruction(question_data):
    answer = question_data["answer"]
    quest_type = question_type(question_data)
    if quest_type == "mcq":
        base_instruction = "From now on, you will act as a teacher in a science and engineering university. When a student asks you a question, you must provide all the reasoning steps before answering the question, you are forbidden to give away the answer before explaining your reasoning. Questions are multiple choice questions with possibly multiple correct answers"
    
    elif quest_type == "simple":
        base_instruction = "From now on, you will act as a teacher in a science and engineering university. When a student asks you a question, you must provide all the reasoning steps before answering the question, you are forbidden to give away the answer before explaining your reasoning."

    answer_prompt = "\nYou must give your reasoning and answer given that the correct answer is: {}".format(answer)
    explanation = question_data.get("explanation", None)
    explanation_prompt = ""
    if explanation is not None:
        explanation_prompt = "\nHere is the explanation of the solution: {}".format(explanation)

    instruction = base_instruction + answer_prompt + explanation_prompt
    return instruction

### SESSION FUNCTIONS

def set_api_key(args):
    confirmation = "n"
    key = read_api_key(args)
    print(f"Your sessions will use API key: {key}")
    confirmation = input("Do you confirm? [y]/n\n")
    if confirmation not in AFFIRMATION_TEXT:
        print("Cancelled.")
        exit(0)
    gpt_wrapper.api_key = key
    print("API key set.")

def get_session_name(args, question_data):
    session_name = args.get("--session-basename", f"test-0")
    if session_name == "auto":
        session_name = f"chat:question-{question_data['sol_id']}"
    return session_name

### PROMPT BUILDING AND CONFIDENCE FUNCTIONS

def question_type(question_data):
    if question_data.get("choices", None) is None:
        return "simple"
    return "mcq"

def assemble_mcq_question(question_data):
    question = question_data["question"]

    # create 4 questions format and pick one at random
    format_chosen = random.randint(0, 3)

    if format_chosen == 0:
        # use numbers
        for i, choice in enumerate(question_data["choices"]):
            question += f"\n{i+1}) {choice}"
    elif format_chosen == 1:
        # use letters
        for i, choice in enumerate(question_data["choices"]):
            question += f"\n{chr(ord('a') + i)}) {choice}"
    elif format_chosen == 2:
        # use roman numerals
        for i, choice in enumerate(question_data["choices"]):
            question += f"\n{roman.toRoman(i+1)}) {choice}"
    elif format_chosen == 3:
        # use bullet points
        for i, choice in enumerate(question_data["choices"]):
            question += f"\n• {choice}"

    return question

def assemble_simple_question(question_data):
    return question_data["question"]

def assemble_question(question_data, 
                      q_type_to_func={
                          "mcq": assemble_mcq_question, 
                          "simple": assemble_simple_question}):
    q_type = question_type(question_data)
    return q_type_to_func[q_type](question_data)

def check_instruct_mode(mode, supported=set(["manual", "random"])):
    if mode not in supported:
        raise ValueError(f"{mode} is not supported.")
    return mode



### LOGGING FUNCTIONS

def create_log_a_q(chat_id: int, solution_id: int, instruction: str, query_texts: str, reply_texts: str):
    log = {
        "sol_id": solution_id,
        "chat_id": chat_id,
        "interaction": []
    }
    
    # append the instruction if not empty
    if len(instruction) > 0:
        log["interaction"].append({"role": "system", "content": instruction})
    
    for query, reply in zip(query_texts, reply_texts):
        log["interaction"].append({"role": "user", "content": query})
        log["interaction"].append({"role": "assistant", "content": reply})

    return log

def create_log_na_q(course_id: int, question_id: int, question: str, message: str ="Non-answerable Question"):
    log = {
        "course_id": course_id,
        "question_id": question_id,
        "chat_id": None,
        "confidence": None,
        "interaction": None,
        "error": f"{message}: {question}"
    }
    return log

############################################
# MAIN
############################################

### call with: python3 interact_with_gpt.py <args-filename>

if __name__ == "__main__":
    # append sys path
    sys.path.append("..")
    
    # read and parse command line arguments
    argsfilename = sys.argv[1]
    args = read_args(argsfilename)
    
    # debug mode?
    debug = args.get("--debug", None)
    debug = debug is not None
    if debug: print("Entering DEBUG mode...")
    
    # resume?
    resume_qid = args.get("--resume-from", None)
    resume = resume_qid is not None
    if resume_qid is not None:
        resume_qid = int(resume_qid)
    
    
    # seeding
    rand.seed(int(42))
    
    # get questions and instruction templates
    a_questions_data = read_questions_data(args)
        
    # setup
    set_api_key(args)
    init_budget = Chat.budget()
    print(f"You budget is [{init_budget['limit'] - init_budget['usage']}/{init_budget['limit']}] tokens.")
    
    ck_filename = args.get("--ck-file", None)
    checkpointing = ck_filename is not None
    print("Checkpointing is", "enabled." if checkpointing else "disabled.")
    
    # start the interaction
    logs = []
    log_filename = args.get("--log-file", "m1_submission_{sciper}.json")
    
    
    # resuming from a specific question
    # resuming from a specific question
    if resume:
        for i, question_data in enumerate(a_questions_data):
            resume_from = i
            if question_data["sol_id"] == resume_qid:
                break
        a_questions_data = a_questions_data[resume_from:]
    
    # interaction for answerable questions
    for i, question_data in enumerate(tqdm(a_questions_data)):
        if question_data.get("answer", None) is None:
            print("skipped question", question_data["sol_id"])
            continue
        # create a new chat session
        session_name = get_session_name(args, question_data)
        #print("Session: ", session_name)
        chat = Chat.create(session_name)
        chat_id = chat.to_dict()["chat_id"]
        
        # start the interaction
        #print("Starting a new interaction...")
        
        # question
        #course_id = question_data["course_id"]
        solution_id = question_data["sol_id"]
        content = assemble_question(question_data)
        #print(f"Q: {content}")
        
        # instruction
        instruction = define_instruction(question_data)
        #print(f"I: {instruction}")
        
        # prompt the assistant
        query_texts, reply_texts = [], []
        # collect the content
        query_texts.append(content)
            
        # interact with the assistant
        if len(instruction) > 0:
            reply_data = chat.ask(content=content, instruction=instruction, model_args=MODEL_ARGS)
        else:
            #reply_data = chat.ask(content=content, model_args=MODEL_ARGS)
            raise ValueError("No instruction provided.")
        
        reply_data = reply_data.to_dict()
        #print(f"A: {reply_data['content']}")
            
        
        # collect the reply
        reply_texts.append(reply_data["content"])
            
        # logging the chat
        #print("Collecting the chat log...")
        log = create_log_a_q(chat_id, solution_id, instruction, query_texts, reply_texts)
        logs.append(log)
        if checkpointing:
            with jsonlines.open(ck_filename, 'a') as jsonl_writer:
                jsonl_writer.write(log)
        #print("done.")
        
        if debug: break #DEBUG  
        
    print("Writing all logs...")
    with open(log_filename, 'w') as json_file:
        if checkpointing: 
            with jsonlines.open(ck_filename, 'r') as jsonl_reader:
                logs = [log for log in jsonl_reader.iter(type=dict)]
                json.dump(logs, json_file, indent=2)
        else:
            json.dump(logs, json_file, indent=2)
    print("done.")
    
    new_budget = Chat.budget()
    print(f"You have consumed {new_budget['usage'] -  init_budget['usage']} tokens.")
    print(f"Your new budget is [{new_budget['limit'] - new_budget['usage']}/{new_budget['limit']}] tokens.")