import json
import os
from mutations import mutate
from llm_utils import summarize_with_prompt
from evaluation import evaluate_summary

MUTATION_NAMES = [
    "synonym_replacement",
    "prompt_rewriting",
    "style_instruction",
    "audience_information",
    "stepwise_prompt"
]

def _normalize_population(initial_prompts):
    norm = []
    for i, p in enumerate(initial_prompts):
        if isinstance(p, dict):
            name = p.get("name", f"p{i+1}")
            text = p.get("text", "")
        else:
            name = f"p{i+1}"
            text = str(p)
        norm.append({"name": name, "text": text})
    return norm

def run_evolution(datasets, initial_prompts, max_rounds=5, top_k=5, rouge1_threshold=0.5):
    population = _normalize_population(initial_prompts)
    RESULT_DIR = "results"
    os.makedirs(RESULT_DIR, exist_ok=True)
    meet_file = os.path.join(RESULT_DIR, "meet_threshold.json")
    selected_file = os.path.join(RESULT_DIR, "selected_topk.json")
    open(meet_file, "w").close()
    open(selected_file, "w").close()

    for round_idx in range(1, max_rounds + 1):
        print(f"\n=== Round {round_idx} | candidates={len(population)} ===")
        candidates = []
        for parent in population:
            candidates.append({
                "round": round_idx,
                "parent_name": parent["name"],
                "mutation": "none", 
                "prompt_text": parent["text"]
            })
            for mtype in MUTATION_NAMES:
                mutated_list = mutate(parent["text"], mtype)
                for mp in mutated_list:
                    candidates.append({
                        "round": round_idx,
                        "parent_name": parent["name"],
                        "mutation": mtype,
                        "prompt_text": mp
                    })
        print(f"[round {round_idx}] candidates={len(candidates)}")

        results = []
        total = sum(len(samples) * len(candidates) for samples in datasets.values())
        count = 0
        for dataset_name, samples in datasets.items():
            for item in samples:
                article_id = item["id"]
                article = item["article"]
                ref = item["reference"]
                for cand in candidates:
                    summary = summarize_with_prompt(article, cand["prompt_text"])
                    scores = evaluate_summary(summary, ref, article)
                    record = {
                        "round": cand["round"],
                        "dataset": dataset_name,
                        "article_id": article_id,
                        "prompt_name": cand["parent_name"],
                        "mutation": cand["mutation"],
                        "prompt_text": cand["prompt_text"],
                        "summary": summary,
                        "rouge1": scores["rouge1"],
                        "rougel": scores["rougel"],
                        "fre": scores["fre"],
                        "compression": scores["compression"]
                    }
                    results.append(record)
                    count += 1
                    if count % 10 == 0:
                        print(f"[Round {round_idx}] Progress: {count}/{total} ({count/total:.1%})")

        print(f"[round {round_idx}] results={len(results)}")

        with open(os.path.join(RESULT_DIR, f"round_{round_idx}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        meet = [r for r in results if r["rouge1"] >= rouge1_threshold]
        remain = [r for r in results if r["rouge1"] < rouge1_threshold]

        if meet:
            with open(meet_file, "a", encoding="utf-8") as f:
                for item in meet:
                    item["meet_round"] = round_idx
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        remain.sort(key=lambda x: x["rouge1"], reverse=True)
        selected = remain[:top_k]

        if selected:
            with open(selected_file, "a", encoding="utf-8") as f:
                for item in selected:
                    record = {
                        "selected_round": round_idx,
                        "prompt_name": item["prompt_name"],
                        "mutation": item["mutation"],
                        "rouge1": item["rouge1"],
                        "prompt_text": item["prompt_text"],
                        "dataset": item["dataset"],
                        "article_id": item["article_id"]
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        population = [{"name": s["prompt_name"], "text": s["prompt_text"]} for s in selected]

if __name__ == "__main__":
    with open("data/cnn_input.json", "r", encoding="utf-8") as f:
        cnn_data = json.load(f)
    with open("data/xsum_input.json", "r", encoding="utf-8") as f:
        xsum_data = json.load(f)

    datasets = {
        "cnn": [{"id": item["id"], "article": item["article"], "reference": item["reference"]} for item in cnn_data],
        "xsum": [{"id": item["id"], "article": item["article"], "reference": item["reference"]} for item in xsum_data]
    }

    INITIAL_PROMPTS = [
        {"name": "zero_shot", "text": "Can you give me a short summary of this article?"},
        {"name": "few_shot", "text": """Here are some examples of articles and their summaries:

        Example 1:
        Article: Anthony Doerr's All the Light We Cannot See, a novel centered on the World War II bombing of St.-Malo, France, and two characters on opposite sides of the war, won the Pulitzer Prize for fiction Monday. Doerr's novel had received rave reviews upon its release last spring. \"I must blame Anthony Doerr for lost sleep, because once I started reading his new novel, 'All the Light We Cannot See,' there was no putting it down,\" wrote William T. Vollmann in The New York Times Book Review. Doerr's work was also a finalist for the National Book Award. It's his second novel and fourth work of fiction, including two short story collections. 2015 Pulitzer Prize winners in journalism named . \"Between Riverside and Crazy,\" a play by Stephen Adly Guirgis, won the Pulitzer for drama. An earlier Guirgis work, \"The Motherf***** with the Hat,\" ran on Broadway in 2011. Elizabeth Kolbert's \"The Sixth Extinction: An Unnatural History\" won the Pulitzer for general nonfiction. Kolbert, a New Yorker staff writer, tackles the idea that we're at the beginning of another mass die-off. \"As the planet warms up, and carbon dioxide acidifies the oceans, all bets are off -- except the ones hinging on mass extinctions,\" wrote Nicholas Lazard in The Guardian. Despite that prospect, he added, \"Kolbert's book is not, thankfully, as depressing as you might think. She has a good grip on her subject and uses a light touch when it is most needed.\" Other winners in arts and letters categories include \"Encounters at the Heart of the World: A History of the Mandan People\" by Elizabeth A. Fenn (history); \"The Pope and Mussolini: The Secret History of Pius XI and the Rise of Fascism in Europe\" by David I. Kertzer (biography/autobiography); \"Anthracite Fields\" by Julia Wolfe (music); and \"Digest\" by Gregory Pardlo (poetry). The Pulitzer Prizes are administered by Columbia University and are considered some of the most prestigious honors in journalism and literature.
        Summary: Anthony Doerr's All the Light We Cannot See wins Pulitzer for fiction . Elizabeth Kolbert's The Sixth Extinction wins general nonfiction prize .

        Example 2:
        Article: Fit-again midfielder Jimmy Ryan had instant impact, setting up Ash Hunter to lob in the opener early on.\nWithin minutes it was 2-0, with Bobby Grant nodding home Antoni Sarcevic's cross from the left.\nIn the second half, Scunthorpe's Paddy Madden scored from the spot when Tom Hooper was pushed over, but Fleetwood ended strongly for a deserved win.\nScunthorpe caretaker boss Nick Daws told BBC Radio Humberside:\nMedia playback is not supported on this device\n\"I think the performance was probably a bit chalk and cheese in terms of the two halves of the game.\n\"Everybody knows that getting the first goal is key, we didn't get that.\n\"We expected a really tough challenge after a really long week and successful week for us, but they got the first goal.\n\"Two things in the first half put us on the back foot, but we adjusted the shape slightly to stay in the game - I thought we did that really well and that was the foundation for our second-half performance which was excellent.
        Summary: Fleetwood boosted their survival hopes by beating Scunthorpe for just their second League One win of the year.

        Now here is a new article. Please write its summary:"""},
        {"name": "instruction_based", "text": "Please explain the main points of this article briefly in three sentences: "},
        {"name": "pattern_based", "text": "Can you list the three most important facts from this article as bullet points?"},
        {"name": "target_audience", "text": "Can you rewrite this article so that it’s easy to understand for everyday readers?"}
    ]

    run_evolution(datasets, INITIAL_PROMPTS, max_rounds=5, top_k=5, rouge1_threshold=0.5)
