#Evaluation Module
from difflib import SequenceMatcher


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def evaluate_system(agentic_reasoner, eval_data):

    scores = []

    for row in eval_data:

        result = agentic_reasoner.answer(row["question"])

        score = similarity(
            result["answer"].lower(),
            row["ground_truth"].lower()
        )

        scores.append(score)

        print("\nQUESTION:", row["question"])
        print("ANSWER:", result["answer"])
        print("EXPECTED:", row["ground_truth"])
        print("SCORE:", score)

    avg_score = sum(scores) / len(scores)

    print("\nAverage Score:", avg_score)

    return avg_score