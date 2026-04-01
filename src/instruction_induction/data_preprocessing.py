import pandas as pd
import json
import random


def data_processing(path,
                    few_shot_size,
                    eval_size,
                    save_path):
    raw_data = pd.read_excel(path, engine='openpyxl', index_col=0).reset_index(drop=True)

    overall_indices = {
        score: raw_data.index[raw_data['Overall'] == score].tolist()
        for score in raw_data['Overall'].unique()
    }
    
    ii_list = shots_sampling(overall_indices, few_shot_size)
    create_few_shot_sets(ii_list, raw_data, save_path)
    
    all_indices = list(range(len(raw_data)))
    flat_used_indices = set(sum(ii_list, []))
    remaining_indices = list(set(all_indices) - flat_used_indices)
    
    sampled_sets, remaining_indices_after_sampling = random_sampling(remaining_indices, eval_size, 2)
    create_eval_sets(sampled_sets, raw_data, save_path)
    

def shots_sampling(overall_indices, set_num):
    sample_list = []
    
    for _ in range(set_num):
        sample = []
        for score, indices in overall_indices.items():
            if len(indices) > 0:
                selected = random.sample(indices, 1)[0]
                sample.append(selected)
                indices.remove(selected)
        
        sample_list.append(sample)
    
    return sample_list

def create_few_shot_sets(idx_list, raw_data, save_path):
    few_shot_sets = {}
    for i_idx, ii in enumerate(idx_list, start=1):
        few_shot_set = []
        for idx in ii:
            few_shot_data = (
                f"Question : {raw_data['Question'][idx]}\n"
                + f"Answer_Essay : {raw_data["Essay"][idx]}\n"
                + f"Score : {raw_data['Overall'][idx]}"
            )
            few_shot_set.append(few_shot_data)
        few_shot_sets[f"{i_idx}"] = "\n\n".join(few_shot_set)
        
    with open(save_path+"/few_shot_sets.json", 'w', encoding='utf-8') as f:
        json.dump(few_shot_sets, f, indent=4)
        
def create_eval_sets(sampled_sets, raw_data, save_path):
    eval_sets = {}
    for set_idx, eval_set in enumerate(sampled_sets, start=1):
        eval_data = {"Question":[],
                    "Essay":[],
                    "Overall":[]
                    }
        for i in eval_set:
            eval_data["Question"].append(raw_data['Question'][i])
            eval_data["Essay"].append(raw_data['Essay'][i])
            eval_data["Overall"].append(int(raw_data['Overall'][i]))

        eval_sets[f"eval_{set_idx}"]=eval_data
               
    with open(save_path+"/eval_sets.json", 'w', encoding='utf-8') as f:
        json.dump(eval_sets, f, indent=4)

def random_sampling(remaining_indices, sample_size, sample_num):
    sampled_sets = []
    for _ in range(sample_num):
        sample = random.sample(remaining_indices, sample_size)
        sampled_sets.append(sample)
        remaining_indices = list(set(remaining_indices) - set(sample))
    return sampled_sets, remaining_indices

if __name__ == "__main__":
    class CONFIG:
        path="./data/raw/ielts/ielts_data.xlsx"
        few_shot_size=20
        eval_size= 100
        save_path="./data/input/ielts"
        
    data_processing(CONFIG.path,
                    CONFIG.few_shot_size,
                    CONFIG.eval_size,
                    CONFIG.save_path)