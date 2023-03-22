import json
import pandas as pd

if __name__ == '__main__':
    print("Parse datasets to statements")
    exp_val = pd.read_csv("./data/explagraphs/dev_original.tsv", sep="\t", header=0)
    exp_train = pd.read_csv("./data/explagraphs/train_original.tsv", sep="\t", header=0)
    exp_train.columns = ["belief", "argument", "label", "gold_graph"]
    exp_val.columns = ["belief", "argument", "label", "gold_graph"]
    exp_df = pd.concat([exp_train, exp_val], axis=0)
    exp_df['id'] = range(880, 880+len(exp_df))
    copa_dev = pd.read_json("./data/copa/copa_dev_original.jsonl", lines=True)
    copa_test = pd.read_json("./data/copa/copa_test_original.jsonl", lines=True)
    copa_df = pd.concat([copa_dev, copa_test], axis=0)

    with open('expla_statements.jsonl', 'a') as the_file:
        for index, row in exp_df.iterrows():
            r = {'id': row['id'], 'question': {'choices': [{'text': row['belief']}]}, 'statements': [{'statement': row['argument']}]}
            the_file.write(json.dumps(r) + "\n")

    with open('copa_statements.jsonl', 'a') as the_file:
        for index, row in copa_df.iterrows():
            s1 = row['p'] + " " + row['a1']
            s2 = row['p'] + " " + row['a2']

            r = {'id': row['id'], 'question': {'choices': [{'text': s1}]}, 'statements': [{'statement': s2}]}
            the_file.write(json.dumps(r) + "\n")
