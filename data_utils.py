from datasets import load_dataset

def load_datasets(names, n_samples=2):
    datasets = {}
    for name in names:
        ds = load_dataset(name, split="test[:{}]".format(n_samples))
        if name == "cnn_dailymail":
            datasets[name] = [(x["article"], x["highlights"]) for x in ds]
        elif name == "EdinburghNLP/xsum":
            datasets[name] = [(x["document"], x["summary"]) for x in ds]
    return datasets
