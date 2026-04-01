# python -m src.instruction_induction.induction
import json
import ollama

def istruction_induction(few_shot_sets_path, ollama_model, save_path):
    initial_instructions = {}
    with open(few_shot_sets_path, 'r', encoding='utf-8') as f:
        few_shot_sets = json.load(f)
    
    for idx in few_shot_sets:
        few_shot = few_shot_sets[idx]
        response =generate_instruction(few_shot, ollama_model)
        initial_instructions[idx] = response

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(initial_instructions, f, indent=4)
    
def generate_instruction(few_shot, ollama_model):
    prompt = f'''I gave an IELTS Writing examiner an Instruction and Inputs that are Question-Answer Essay pairs. 
The examiner read the instruction and scored every Input answer essay.

Here are the Input Question-Answer pairs and Score:

{few_shot}

Based on these Input pairs and Scores, please infer the instruction that was given to the examiner. Infer instructions without further explanation.'''
    response = ollama.chat(
        model=ollama_model,
        messages=[
            {"role": "system", "content": "You are a prompt Engineer. Generate the instruction without further explanation."},
            {'role': 'user', 'content': prompt}],
        options={"num_keep": 100,
                 },
        )
    return response['message']['content']
    

if __name__ == "__main__":
    class CONFIG:
        few_shot_sets_path="./data/input/ielts/few_shot_sets.json"
        ollama_model = "llama3:70b-instruct"
        save_path="./results/ielts/initial_instruction/instruction_pool.json"
    istruction_induction(CONFIG.few_shot_sets_path, CONFIG.ollama_model, CONFIG.save_path)