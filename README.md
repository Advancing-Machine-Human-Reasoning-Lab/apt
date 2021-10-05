# APT

This is the repository based on a paper accepted in ACL 2021: [Improving Paraphrase Detection with the Adversarial Paraphrasing Task](https://aclanthology.org/2021.acl-long.552/).

## Repository vs Paper
- APP is the human split of the adversarial dataset (AP_H)
- NAP is the neural network's attempt at the APT (AP_T5)

## Packages needed:
All the packages used in this repository are listed below linked to their installation instructions. However, you might not need to install all if you want to run only some of the functionalities of the repository (for instance, you would not need any of the flask packages if you do not want to run the web-based APT). Please check the import statements in the scripts you want to run before installing packages to avoid installing unnecessary packages.
- [bleurt](https://github.com/google-research/bleurt)
- [transformers](https://huggingface.co/transformers/installation.html)
- [torch](https://pytorch.org/get-started/locally/)
- [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [numpy](https://numpy.org/install/)
- [matplotlib](https://matplotlib.org/stable/users/installing.html)
- [tqdm](https://pypi.org/project/tqdm/)
- [flask](https://flask.palletsprojects.com/en/2.0.x/installation/)
- [flask-cors](https://flask-cors.readthedocs.io/en/latest/)
- [flask-session](https://flask-session.readthedocs.io/en/latest/)
- [waitress](https://pypi.org/project/waitress/)

## Main scripts in this repository
Here is a list of all scripts in this repository along with a brief description of what they do:
- `apt.py` runs the web-based APT. This is what we used for our mTurk study.
- `graph.py` generates the graphs which can be used to compare datasets.
- `nap_generation.py` uses a fine-tuned T5 model to write paraphrases taking source sentences from MSRP and TwitterPPDB. The code to fine-tune the T5 model can be found inside `paraphraser-for-apt/`.

## Model and Datasets
The fine-tuned T5 paraphraser can be accessed using huggingface as follows:
```py
from transformers import T5Tokenizer, T5ForConditionalGeneration
paraphrasing_tokenizer = T5Tokenizer.from_pretrained("t5-base")
paraphrasing_model = T5ForConditionalGeneration.from_pretrained("coderpotter/T5-for-Adversarial-Paraphrasing")

# this function will take a sentence, top_k and top_p values based on beam search
def generate_paraphrases(sentence, top_k=120, top_p=0.95):
    text = "paraphrase: " + sentence + " </s>"
    encoding = paraphrasing_tokenizer.encode_plus(text, max_length=256, padding="max_length", return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    beam_outputs = paraphrasing_model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=top_k,
        top_p=top_p,
        early_stopping=True,
        num_return_sequences=10,
    )
    final_outputs = []
    for beam_output in beam_outputs:
        sent = paraphrasing_tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)
    return final_outputs
```
Please refer to `nap_generation.py` for ways to better utilize this model using concepts of [top-k sampling](https://arxiv.org/abs/1805.04833) and [top-p sampling](https://arxiv.org/abs/1904.09751).

The fine-tuned paraphrase detector can be accessed through huggingface as follows:
```py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("coderpotter/adversarial-paraphrasing-detector")
model = AutoModelForSequenceClassification.from_pretrained("coderpotter/adversarial-paraphrasing-detector")
```

The Adversarial dataset can be found [here](https://drive.google.com/file/d/1a4_w9ZXMoD8AHcnLi6BIHtUCSDhsRlJp/view?usp=sharing).
