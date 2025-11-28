import sys
import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)
# Will be Removed Later
# =====================================================================

import os 
import json 
import torch 
import gc

from LKN.utils import  get_num_layers
from LKN.probers import get_prober
from LKN.utils import train_prober 
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from tqdm import tqdm  
from LKN.utils import set_seed

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in ("true", "t", "1", "yes", "y", "on"):
        return True
    if val in ("false", "f", "0", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value for only_last_token: {value}")

def main(args):
    model_name = args.model_name
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    eval_every = args.eval_every
    prompt_type = args.prompt_type
    args.only_last_token = str_to_bool(args.only_last_token)
    set_seed(args.init_seed)

    train_Y_original = torch.load(f"outputs/activations/{model_name}/{dataset_name}/{prompt_type}/train/labels.pt")
    test_Y_original = torch.load(f"outputs/activations/{model_name}/{dataset_name}/{prompt_type}/test/labels.pt")

    if args.only_last_token:
        train_Y_original = train_Y_original[:, -1]
        test_Y_original = test_Y_original[:, -1]


    config =  OmegaConf.create(vars(args))
    config.save_dir = f"outputs/raw_key_probers/{model_name}/{dataset_name}/{prompt_type}/only_last_token_{args.only_last_token}/{args.init_seed}"
    os.makedirs(config.save_dir, exist_ok=True)
    OmegaConf.save(config, f"{config.save_dir}/config.yaml")

    if os.path.exists(f"{config.save_dir}/layer_0_results.json"):
        print(f"Layer 0 results already exists for {model_name} {dataset_name} {prompt_type} {args.only_last_token} {args.init_seed}")
        exit(0)


    # ------------------------------------------------------------------------------------
    # Load Activations 
    # ------------------------------------------------------------------------------------
    llm_num_layers = get_num_layers(model_name)
    even_index_layers = list(range(0, llm_num_layers, 2)) + [llm_num_layers - 1] if llm_num_layers % 2 == 1 else list(range(0, llm_num_layers, 2))
    pbar = tqdm(even_index_layers, desc=f"{config.save_dir}")
    for target_layer in pbar:
        
        train_X_original = torch.load(f"outputs/activations/{model_name}/{dataset_name}/{prompt_type}/train/layer_{target_layer}_key.pt")
        test_X_original = torch.load(f"outputs/activations/{model_name}/{dataset_name}/{prompt_type}/test/layer_{target_layer}_key.pt")
        hidden_dim= train_X_original.shape[2]

        train_X_original = train_X_original.float()
        test_X_original = test_X_original.float()

        if args.only_last_token:
            train_X = train_X_original[:, -1, :]
            test_X = test_X_original[:, -1, :]
            train_Y = train_Y_original
            test_Y = test_Y_original
        else:
            N1, T1, D1 = train_X_original.shape
            N2, T2, D2 = test_X_original.shape
            train_X = train_X_original.reshape(N1*T1, D1).float()
            test_X = test_X_original.reshape(N2*T2, D2).float()
            train_Y = train_Y_original.reshape(N1*T1)
            test_Y = test_Y_original.reshape(N2*T2)    
        
        value_prober  = get_prober("BCEProber", hidden_dim, num_layers=1, init_seed=args.init_seed)

        prober, losses, train_accs, test_accs, eval_steps, thresholds = train_prober(
                value_prober, 
                train_X, 
                train_Y, 
                test_X, 
                test_Y, 
                epochs=epochs, 
                eval_every=eval_every, 
                batch_size=batch_size, 
                lr=lr,
                verbose=False,
        )
        torch.save(prober, f"{config.save_dir}/layer_{target_layer}_prober.pt") 
        results = {}
        results["losses"] = losses
        results["train_accs"] = train_accs
        results["test_accs"] = test_accs
        results["eval_steps"] = eval_steps
        results["thresholds"] = thresholds
        json.dump(results, open(f"{config.save_dir}/layer_{target_layer}_results.json", "w"))
        
        fig, axes = plt.subplots(1,2, figsize=(7, 3))
        axes[0].plot(losses)
        axes[1].plot(train_accs)
        axes[1].plot(test_accs)
        axes[0].set_title("Train Loss")
        axes[1].set_title("Accuracy")
        axes[1].legend(["Train", "Test"])
        fig.suptitle("Value Prober Results")
        plt.tight_layout()
        plt.savefig(f"{config.save_dir}/layer_{target_layer}_results.png")
        plt.close()
        
        # empty the cache
        torch.cuda.empty_cache() 
        gc.collect()
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--eval_every", type=int, required=True)
    parser.add_argument("--init_seed", type=int, required=True)
    parser.add_argument("--only_last_token", type=str, required=True)
    args = parser.parse_args()
    main(args)




