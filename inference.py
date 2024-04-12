import argparse
from task import WikiBioTask
import torch.nn as nn
import numpy as np

def Show(task, model_path, only_keyword=False, use_penalty=True, add_type=True, use_entropy=True, use_idf=True, gamma=0.9, rho=0.01, low_cpu_mem_usage=True):
    print("Task:", task)
    print("Model Path:", model_path)
    print("Only Keyword:", only_keyword)
    print("Use Penalty:", use_penalty)
    print("Add Type:", add_type)
    print("Use Entropy:", use_entropy)
    print("Use IDF:", use_idf)
    print("Gamma:", gamma)
    print("Rho:", rho)
    print("Low CPU Mem Usage:", low_cpu_mem_usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="wikibio")
    parser.add_argument("--model_path", type=str, default="/weights/llama/hf/")
    parser.add_argument("--only_keyword",action="store_true", default=False)
    parser.add_argument("--use_penalty", action="store_true", default=False)
    parser.add_argument("--add_type", action="store_true", default=False)
    parser.add_argument("--use_entropy", action="store_true", default=False)
    parser.add_argument("--use_idf", action="store_true", default=False)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--low_cpu_mem_usage", action="store_true", default=True)
    args = parser.parse_args()
    if args.task == "wikibio":
        Show(args.task, args.model_path, args.only_keyword, args.use_penalty, args.add_type, args.use_entropy, args.use_idf, args.gamma, args.rho, args.low_cpu_mem_usage)
        outputs =  input("Enter the text \n >>").strip()
        t = WikiBioTask(args)
        outputs = t.add_type(outputs)
        words,losses = t.run_generate(prompt=outputs,gamma = 0.9)
        if '.' in outputs[:-1] or '?' in outputs[:-1] or '!' in outputs[:-1]:
          sentence_loss = t.Text(words,losses)
          print(f"{100 * np.mean(t.Norm(sentence_loss))}%")
        else:
          print(f"{100 * np.mean(t.Norm(losses))}%")
