import json
import ollama
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

def evaluation_2nd(final_insruction_path, eval_path, ollama_model, save_path):
    with open(final_insruction_path, 'r', encoding='utf-8') as f:
        final_insructions = json.load(f)
        
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    eval_questions = eval_data["eval_2"]["Question"][:2]
    eval_essay = eval_data["eval_2"]["Essay"][:2]
    eval_score = eval_data["eval_2"]["Overall"][:2]
    
    prompt_sets = evaluate_prompt(final_insructions, eval_questions, eval_essay)
    result_sets = evaluate_essay(prompt_sets, ollama_model)
        
    qwk_sets = {}

    for idx in result_sets:
        kappa_quadratic = cohen_kappa_score(eval_score,
                                            result_sets[idx],
                                            weights='quadratic',
                                            labels=[5,6,7,8,9])  # IELTS labels=[5,6,7,8,9], ASAP++ labels=[0,1,2,3], ELLIPSE labels=[1,2,3,4,5]
        qwk_sets[idx] = kappa_quadratic
    
    all_results = {}  
    for idx in final_insructions:
        all_results[idx] = {
            "instruction": final_insructions[idx],
            "model_score": result_sets[idx],
            "qwk": qwk_sets[idx]
        }
    
    print(all_results)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
        

def evaluate_prompt(final_insructions, eval_questions, eval_essay):
    prompt_sets = {}

    for instruction_idx in final_insructions:
        input_prompt = []
        
        for question, answer in zip(eval_questions, eval_essay):
            prompt = f"""{final_insructions[instruction_idx]}

    Question: {question}
    Answer Essay: {answer}
    Score: """
            input_prompt.append(prompt)
        
        prompt_sets[instruction_idx] = input_prompt
    return prompt_sets

def evaluate_essay(prompt_sets, ollama_model):
    result_sets = {}
    
    for idx in tqdm(prompt_sets):
        result = []
        for prompt in prompt_sets[idx]:
            response = ollama.chat(
                model=ollama_model,
                messages=[
                    {"role": "system", "content": "You are a helpful chatbot of this scoring task. Generate only an overall band score without further explanation. Generate the band score as an integer."},
                    {'role': 'user', 'content': prompt}],
                )
            result.append(int(float(response['message']['content'])))
            
        result_sets[idx] = result
        
    return result_sets


if __name__ == "__main__":
    class CONFIG:
        final_insruction_path="./results/ielts/final_instruction/instruction_pool_csf.json"
        eval_path="./data/input/ielts/eval_sets.json"
        ollama_model = "llama3:70b-instruct"
        save_path="./results/ielts/final_instruction/results_csf.json"
        
    evaluation_2nd(CONFIG.final_insruction_path, CONFIG.eval_path, CONFIG.ollama_model, CONFIG.save_path)