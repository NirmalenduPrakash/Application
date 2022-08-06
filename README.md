# HatefulMemes

## Steps

1. pip install -r /Hateful meme detection/VL-BERT/requirements.txt

2. sh /Hateful meme detection/VL-BERT/scripts/init.sh

3. Copy pretrained VL bert(from here-https://drive.google.com/file/d/15IAT7NVCXtTj_9itl7OXtA_jXRwiaVWZ/view?usp=sharing) to /Hateful meme detectionVL-BERT/pretrain_model/ 

4. To train, sh /Hateful meme detection/VL-BERT/scripts/dist_run_single.sh 4 /Hateful meme detection/VL-BERT/cls/train_end2end.py "/Hateful meme detection/VL-BERT/cfgs/cls/base_4x14G_fp32_k8s_v4.yaml" "/Hateful meme detection/VL-BERT/pretrain_model/vl-bert/vl-bert-base-e2e"

5. Alternately, you can download pretrained hatefull model from here-

5. To test, python /Hateful meme detection/VL-BERT/cls/test.py --cfg "/Hateful meme detection/VL-BERT/cfgs/cls/base_4x14G_fp32_k8s_v4.yaml" --ckpt /Hateful meme detection/VL-BERT/pretrain_model/vl-bert/vl-bert-base-e2e/base_4x14G_fp32_k8s_v4/train_train/vl-bert_base_res101_cls-0049.model --bs 4 --gpus 0 --result-path [result path] --result-name "[file name for result file]"

6. Instruction to run ERNIE and UNITER available in /Hateful meme detection/README

7. To run the priliminary study to identify count of abstractive and metaphorical memes, install following:
pip install joblib
pip install nltk
pip install git+https://github.com/openai/CLIP.git
Copy conceptual_weights.pt(download from here) in the root directory
RUN python -m spacy download en
RUN python -m nltk.download wordnet
python /Hateful meme detection/get_match_count.py

Hateful meme detection code from https://github.com/HimariO/HatefulMemesChallenge
Caption generation code from https://github.com/rmokady/CLIP_prefix_caption
Emotion generation code from https://github.com/twitter-research/visual-sentiment-analysis (alread run and added emotion tags in //Hateful meme detection/data/box_annos_race_1.json)

# Text Summarization

## Steps

1. Download the processed input for cnndm as mentioned in /TextSummarization/README

2. pip install -r /TextSummarization/requirements.txt

3. pip install compare-mt

4. To train, run: python /TextSummarization/main.py --cuda --gpuid 0 --config cnndm -l (i have not yet figured how to run this on multi gpus)

5. To eval, follow /TextSummarization/README

