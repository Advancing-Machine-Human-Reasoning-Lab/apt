from transformers import T5ForConditionalGeneration

paraphrasing_model = T5ForConditionalGeneration.from_pretrained("paraphraser-for-apt/t5_paraphrase1/model2")
paraphrasing_model.push_to_hub("T5-for-Adversarial-Paraphrasing")