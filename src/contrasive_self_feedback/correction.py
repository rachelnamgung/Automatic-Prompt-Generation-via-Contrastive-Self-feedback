import json
import ollama

def extract_bad_instruction(initial_data, ratio):
    sorted_items = sorted(initial_data.items(), key=lambda x: x[1]['qwk'], reverse=True)
    count = int(len(sorted_items) * ratio)
    worst_idx = [item[0] for item in sorted_items[-count:]]
    return worst_idx

def making_contrastive_samples(initial_data, worst_idx, eval_questions, eval_essay, eval_score):
    contrastive_samples_set = {}
    for idx in worst_idx:
        model_score = initial_data[idx]["model_score"]
        contrastive_samples = []
        
        for i in range(len(eval_score)):

            if model_score[i] != eval_score[i]:
                contrastive_sample = f"- Question: {eval_questions[i]}\n- Essay: {eval_essay[i]}\n- Wrong score: {model_score[i]}\n- Actual score: {eval_score[i]}"
                contrastive_samples.append(contrastive_sample)
        contrastive_samples_set[idx]={}
        contrastive_samples_set[idx]["instruction"] = initial_data[idx]["instruction"]
        contrastive_samples_set[idx]["contrastive_samples"] = "\n\n".join(contrastive_samples)
    
    return contrastive_samples_set

def making_cs_prompt(criteria_path, contrastive_samples_set):
    with open(criteria_path, "r", encoding="utf-8") as f:
        template = f.read()
        
    self_corr_prompts = {}
    for idx in contrastive_samples_set:
        instruction = contrastive_samples_set[idx]["instruction"]
        contrastive_samples = contrastive_samples_set[idx]["contrastive_samples"]
        prompt = template.format(instruction=instruction,
                                contrastive_samples=contrastive_samples)
        self_corr_prompts[idx] = prompt
    
    return self_corr_prompts
    

def contrastive_correction(initial_result_path, eval_path, criteria_path, ratio, ollama_model, save_path):
    with open(initial_result_path, 'r', encoding='utf-8') as f:
            initial_data = json.load(f)
            
    worst_idx = extract_bad_instruction(initial_data, ratio)
    
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
        
    eval_questions = eval_data["eval_1"]["Question"][:2]
    eval_essay = eval_data["eval_1"]["Essay"][:2]
    eval_score = eval_data["eval_1"]["Overall"][:2]

    contrastive_samples_set = making_contrastive_samples(initial_data,
                                                         worst_idx,
                                                         eval_questions,
                                                         eval_essay,
                                                         eval_score)

        
    self_corr_prompts = making_cs_prompt(criteria_path, contrastive_samples_set)
        
    # self_corrected = {}

    # # for idx in self_corr_prompts:
        
    # #     response = ollama.chat(
    # #         model=ollama_model,
    # #         messages=[
    # #         {"role": "system", "content": "You are a scoring rubric expert. Improve the instruction to align with the official Essay Scoring Guidelines. Be specific, objective, and actionable. Output only the revised instruction."},
    # #         {'role': 'user', 'content': self_corr_prompts[idx]}],
    # #         )
    # #     print(response['message']['content'])
    # #     self_corrected[idx] = response['message']['content']
        
    self_corrected = {'5': "test1",'6': "test2",'7': "test3",'8': "test4",'9': "test5"}

    instruction_pool_cs = {}
    
    for idx in initial_data:
        if idx in list(self_corrected.keys()):
            instruction_pool_cs[idx]=self_corrected[idx]
        else:
            
            instruction_pool_cs[idx]=initial_data[idx]["instruction"]
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(instruction_pool_cs, f, indent=4)

if __name__ == "__main__":
    class CONFIG:
        initial_result_path="./results/ielts/initial_instruction/results.json"
        eval_path="./data/input/ielts/eval_sets.json"
        criteria_path="./data/input/ielts/criteria.txt"
        ratio = 0.75
        ollama_model = "llama3:70b-instruct"
        save_path="./results/ielts/final_instruction/instruction_pool_csf.json"
        
    contrastive_correction(CONFIG.initial_result_path, CONFIG.eval_path, CONFIG.criteria_path, CONFIG.ratio, CONFIG.ollama_model, CONFIG.save_path)