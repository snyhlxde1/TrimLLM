import inspect
from lm_eval.base import Task, PerplexityTask, rf
from lm_eval.metrics import mean

def build_deepseek_instruction_prompt(instruction: str):
    return '''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

class PythonPreplexityTask(PerplexityTask):
    VERSION = 0
    DATASET_PATH = "iamketan25/python-qa-instructions-dataset"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        for doc in self.dataset["train"]:
            yield build_deepseek_instruction_prompt(doc["prompt"]) + doc['chosen']

    def test_docs(self):
        for doc in self.dataset["test"]:
            yield build_deepseek_instruction_prompt(doc["prompt"]) + doc['chosen']