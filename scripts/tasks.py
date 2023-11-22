import re

# common-sense QA benchmarks
class PIQA:
    def __init__(self):
        self._template = "Question: {}\nAnswer:"

    def get_context(self, examples):
        ctx = examples['goal']
        return [self._template.format(c) for c in ctx]

    def get_target(self, examples):
        if -1 in examples["label"]:  # test set
            return [""] * len(examples["label"])
        else:
            gt_tuples = [("sol{}".format(label + 1), idx)
                         for idx, label in enumerate(examples['label'])]
            return [examples[k][i] for k, i in gt_tuples]


class HellaSwag:
    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def get_context(self, examples):
        ctx_zip = zip(examples["activity_label"],
                      examples["ctx_a"], examples["ctx_b"])
        return [self.preprocess(a + ": " + b + " " + c.capitalize()) for a, b, c in ctx_zip]

    def get_target(self, examples):
        labels = examples["label"]
        endings = examples["endings"]
        targets = []
        for idx, label in enumerate(labels):
            target = '' if label == '' else endings[idx][int(label)]
            targets.append(self.preprocess(target))
        return targets


class OpenBookQA:
    def get_context(self, examples):
        return examples['question_stem']

    def get_target(self, examples):
        choices = examples['choices']
        answers = examples['answerKey']
        targets = []
        for choice, answer in zip(choices, answers):
            answer = ord(answer.strip()) - ord('A')
            targets.append(choice['text'][answer])
        return targets


class ARC:
    def __init__(self):
        self._template = "Question: {}\nAnswer:"

    def get_context(self, examples):
        ctx = examples['question']
        return [self._template.format(c) for c in ctx]

    def get_target(self, examples):
        choices = examples['choices']
        answers = examples['answerKey']
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        for idx, answer in enumerate(answers):
            answer = num_to_letter.get(answer, answer)
            answer = ord(answer) - ord("A")
            answers[idx] = choices[idx]["text"][answer]
        return answers


class RACE:
    @classmethod
    def doc_to_text(cls, article, question):
        text = "Article: " + article + "\n\n"
        text += "Question: " + question + "\n\n"
        text += "Answer:"
        return text

    def get_context(self, examples):
        return [
            self.doc_to_text(article, question)
            for article, question in zip(examples["article"], examples["question"])
        ]

    def get_target(self, examples):
        answers = examples['answer']
        options = examples['options']
        for idx, answer in enumerate(answers):
            answers[idx] = options[idx][ord(answer) - ord("A")]
        return answers


class SciQ:
    def __init__(self):
        self._template = "{}\nQuestion: {}\nAnswer:"

    def get_context(self, examples):
        sources = examples['support']
        queries = examples['question']
        return [self._template.format(s, q) for s, q in zip(sources, queries)]

    def get_target(self, examples):
        return examples['correct_answer']


class WebQs:
    def get_context(self, examples):
        return ["Question: " + question + "\nAnswer:" for question in examples["question"]]

    def get_target(self, examples):
        return [" " + answers[0] for answers in examples["answers"]]


# medical benchmarks
class MedMCQA:
    def __init__(self):
        self._template = "Question: {}\nAnswer: "
    def get_context(self, examples):
        ctx = examples['question']
        return [self._template.format(c) for c in ctx]
    def get_target(self, examples):
        if -1 in examples["cop"]:  # test set
            return [""] * len(examples["cop"])
        else:
            choices = {}
            choices['0'] = examples['opa']
            choices['1'] = examples['opb']
            choices['2'] = examples['opc']
            choices['3'] = examples['opd']
            options = examples['cop']

            targets = []
            for idx, option in enumerate(options):
                option = str(option)
                correct_choice = choices[option][idx]
                targets.append(correct_choice)
            return targets

# legal benchmarks
class LEXGLUE_casehold:
    def get_context(self, examples):
        return examples['question']
    def get_targets(self, examples):
        choices = examples['endings']
        labels = examples['label']

        targets = []
        for idx, label in enumerate(labels):
            label = ord(label) - ord('0')
            correct_choice = choices[idx][label]
            targets.append(correct_choice)
        return targets

# financial benchmarks
class FiQA:
    def __init__(self):
        self._template = "Question: {}\nAnswer: "
    def get_context(self, examples):
        ctx = examples['question']
        return [self._template.format(c) for c in ctx]
    def get_targets(self, examples):
        answers = examples['ground_truths']

        targets = []
        for a in answers:
            targets.append(a[0])
        return targets
    
class FinanceAlpaca:
    def __init__(self):
        self._template = "Question: {}\nAnswer: "
    def get_context(self, examples):
        ctx = examples['instruction']
        return [self._template.format(c) for c in ctx]
    def get_targets(self, examples):
        answers = examples['output']

        targets = []
        for a in answers:
            targets.append(a[0])
        return targets


task_dict = {
    "piqa": PIQA(),
    "hellaswag": HellaSwag(),
    "openbookqa": OpenBookQA(),
    "arc_easy": ARC(),
    "arc_challenge": ARC(),
    "sciq": SciQ(),
    "web_questions": WebQs(),
    "race": RACE(),
    "medmcqa": MedMCQA(),
    "lexglue_casehold": LEXGLUE_casehold(),
    "financeqa": FinanceAlpaca(),
}


def map_dataset_name_and_config(args):
    dataset_name = args.dataset_name
    dataset_config_name = args.dataset_config_name
    if args.dataset_name == 'arc_easy':
        dataset_name = 'ai2_arc'
        dataset_config_name = 'ARC-Easy'
    elif args.dataset_name == 'arc_challenge':
        dataset_name = 'ai2_arc'
        dataset_config_name = 'ARC-Challenge'
    elif args.dataset_name == 'race':
        dataset_config_name = 'high'
    elif args.dataset_name == 'lexglue_casehold':
        dataset_name == 'lex_glue'
        dataset_config_name == 'case_hold'
    elif args.dataset_name == 'financeqa':
        dataset_name == 'gbharti/finance-alpaca'


    return dataset_name, dataset_config_name


LM_EVAL_TASK_NAME_MAPPING = {
    "web_questions": "webqs"
}
