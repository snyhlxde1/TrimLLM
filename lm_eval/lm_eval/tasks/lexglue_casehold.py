from lm_eval.base import MultipleChoiceTask


class LEX_GLUE_casehold(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "lexglue_casehold"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "question": doc["context"],
            "choices": [doc["endings"][0], doc["endings"][1], doc["endings"][2], doc["endings"][3]],
            "gold": doc["label"],
        }
        return out_doc

    def doc_to_text(self, doc):
        return "Question: " + doc["context"] + "\nAnswer:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]
