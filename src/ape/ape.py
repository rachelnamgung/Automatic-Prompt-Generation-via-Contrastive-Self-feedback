import json
import ollama
# from tqdm import tqdm
# from sklearn.metrics import cohen_kappa_score

def ape_resample(initial_insruction_path, ollama_model, resample_size, save_path):
    with open(initial_insruction_path, 'r', encoding='utf-8') as f:
            initial_data = json.load(f)
            
    sorted_items = sorted(initial_data.items(), key=lambda x: x[1]['qwk'], reverse=True)
    best_idx = [item[0] for item in sorted_items[:resample_size]]
    worst_idx = [item[0] for item in sorted_items[-resample_size:]]

    resample_prompt = []
    resampled_instructions=[]

    for idx in best_idx:
        resample_instruction = f"""Generate a variation of the following instruction while keeping the semantic meaning.
        input: {initial_data[idx]["instruction"]}
        output: """
        resample_prompt.append(resample_instruction)
        
    for resample_instruction in resample_prompt:
        
        response = ollama.chat(
            model=ollama_model,
            messages=[
            {"role": "system", "content": "You are a prompt Engineer. Generate *one* instruction without any other further explanations."},
            {'role': 'user', 'content': resample_instruction}],
            )
        resampled_instructions.append(response['message']['content'])
    
    resampled_instructions = ["test1", "test2", "test3", "test4", "test5"]
    
    for idx, inst in zip(worst_idx, resampled_instructions):
        initial_data[idx]["instruction"] = inst
    
    final_instructions={}
    for idx in initial_data:
        final_instructions[idx] = initial_data[idx]["instruction"]
        
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_instructions, f, indent=4)
    

if __name__ == "__main__":
    class CONFIG:
        initial_insruction_path="./results/ielts/initial_instruction/results.json"
        ollama_model = "llama3:70b-instruct"
        resample_size = 5
        save_path="./results/ielts/final_instruction/instruction_pool_ape.json"
        
    ape_resample(CONFIG.initial_insruction_path, CONFIG.ollama_model, CONFIG.resample_size, CONFIG.save_path)