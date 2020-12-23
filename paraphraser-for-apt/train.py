import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bleurt.score import BleurtScorer

train_df = pd.read_csv("paraphrase_data/train.tsv", sep="\t")
train_df.columns = ["input_text", "target_text"]
eval_df = pd.read_csv("paraphrase_data/val.tsv", sep="\t")
eval_df.columns = ["input_text", "target_text"]

# model_args = Seq2SeqArgs()
# model_args.num_train_epochs = 10
# model_args.no_save = True
# model_args.evaluate_generated_text = True
# model_args.evaluate_during_training = True
# model_args.evaluate_during_training_verbose = True

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-base",  # try t5-base
    args=Seq2SeqArgs(
        output_dir="outputs/",
        overwrite_output_dir=True,
        do_lower_case=False,
        train_batch_size=1,
        eval_batch_size=1,
        num_train_epochs=10,
        no_save=True,
        evaluate_generated_text=True,
        evaluate_during_training=True,
        evaluate_during_training_steps=10000,
        evaluate_during_training_verbose=True,
        fp16=True,
        n_gpu=3,
        save_model_every_epoch=True,
    ),
    early_stopping=True,
    use_cuda=True,
    num_beams=10,
    num_return_sequences=5,
    top_k=120,
    top_p=0.95,
)


bleurt_scorer = BleurtScorer("/home/animesh/MIforSE/bleurt-score/bleurt/bleurt-base-128/")
mi_tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
mi_model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")  # predicts E, N, C


def get_mi_score(s1, s2):  # returns average of s1 and s2
    tokenized_input_seq_pair = mi_tokenizer.encode_plus(s1, s2, max_length=256, return_token_type_ids=True, truncation=True)
    input_ids = torch.Tensor(tokenized_input_seq_pair["input_ids"]).long().unsqueeze(0)
    token_type_ids = torch.Tensor(tokenized_input_seq_pair["token_type_ids"]).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair["attention_mask"]).long().unsqueeze(0)
    outputs = mi_model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=None,
    )
    predicted_probability_12 = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    tokenized_input_seq_pair = mi_tokenizer.encode_plus(s2, s1, max_length=256, return_token_type_ids=True, truncation=True)
    input_ids = torch.Tensor(tokenized_input_seq_pair["input_ids"]).long().unsqueeze(0)
    token_type_ids = torch.Tensor(tokenized_input_seq_pair["token_type_ids"]).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair["attention_mask"]).long().unsqueeze(0)
    outputs = mi_model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=None,
    )
    predicted_probability_21 = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    return int(argmax(predicted_probability_12) == 0 and argmax(predicted_probability_21) == 0)


def get_bleurt(s1, s2):
    return (bleurt_scorer.score([s1], [s2])[0] + bleurt_scorer.score([s1], [s2])[0]) / 2


def count_matches(labels, preds):
    print(labels, preds)
    return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])


# Train the model
model.train_model(train_data=train_df, eval_data=eval_df, show_running_loss=True, matches=count_matches)

# # Evaluate the model
results = model.eval_model(eval_df)

# Use the model for prediction
print(model.predict(["Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."]))
