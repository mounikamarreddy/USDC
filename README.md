# USDC
This repository contains the USDC dataset, and the entire code of **"USDC: A Dataset of User Stance and Dogmatism in Long Conversations"** paper.

# To Download the Dataset
``` bash
pip install data -> just example
```


## USDC Dataset Folder Details
This folder contains the USDC stance and dogmatism datasets. Stance is at post level and given for each comment of the user. Dogmatism is at user level and given for entire user conversation.
There are two sub folders in this folder. The subfolder "USDC Stance Data" contains the "USDC_Stance.pkl" file that contains the information related to the stance classification data and corresponding LLM annotations, the training_stance.pkl file contains the training data and testing_stance.pkl file contains the testing data for Stance classification. 

**The columns in USDC_Stance.pkl file are as follows:**
- **Submission_id** is the unique identifier of the Reddit conversation
- **Subreddit** is the topic name of a Reddit conversation (Eg: Abortion, Guncontrol and so on) 
- **title** is the first post in a Reddit conversation and called as submission post title
- **content** is the description of title, also called as submission body or content
- **reddit_link** is the link to the entire conversation
- **comments** -> Nested replies to the submission
- **author_key** -> unique author identification (tells about author1 or author 2)
- **stance_id** -> unique identifier of comment
- **gpt41106preview_zero_shot_stance_label** -> Stance annotations generated using GPT-4 zero shot prompting
- **gpt41106preview_zero_shot_stance_reason** -> Reasons for selecting the stance label generated using GPT-4 zero shot prompting
- **gpt41106preview_one_shot_stance_label** ->
- **gpt41106preview_one_shot_stance_reason** ->
- **gpt41106preview_few_shot_stance_label**
- **gpt41106preview_few_shot_stance_reason**
- **mistrallargelatest_zero_shot_stance_label**
- **mistrallargelatest_zero_shot_stance_reason**
- **mistrallargelatest_one_shot_stance_label**
- **mistrallargelatest_one_shot_stance_reason**
- **mistrallargelatest_few_shot_stance_label**
- **mistrallargelatest_few_shot_stance_reason**
- **author_key_name** -> corresponding author name of the author key
- **stance_id_timestamp** -> timestamp of the comment/submission
- **stance_id_comment** -> textual content of the stance id (per message)
- **author_names** -> Meta data of authors information for a submission id
- **author_id_details** -> Metadata of authors comments count and ids of the commen
- **majority_vote_stance_label** -> Final gold label for stance classification

The "USDC Dogmatism Data" sub folder contains the information related to the dogmatism data, corresponding LLM annotations and the training and testing data for Dogmatism classification.


