# USDC
This repository contains the USDC dataset, and the entire code of **"USDC: A Dataset of User Stance and Dogmatism in Long Conversations"** paper.

# USDC Dataset Folder
This folder contains the USDC stance and dogmatism datasets.
Stance is at post level and given for each comment of the user.
Dogmatism is at user level and given for entire user conversation.
There are two sub folders in this folder. The first sub folder "USDC Stance Data" contains the information related to the stance classification data, corresponding LLM annotations and the training and testing data for Stance classification. The "USDC Dogmatism Data" sub folder contains the information related to the dogmatism data, corresponding LLM annotations and the training and testing data for Dogmatism classification.

For stance Detection, the columns are as follows:
Submission_id
Subreddit ->
title
content
reddit_link -
'Comments' -> Nested replies to the submission
Author_key -> unique author identification (tells about author1 or author 2)
Stance_id -> unique identifier of comment
Gpt41106preview_zero_shot_stance_label 
gpt41106preview_zero_shot_stance_reason
gpt41106preview_one_shot_stance_label
gpt41106preview_one_shot_stance_reason
gpt41106preview_few_shot_stance_label
gpt41106preview_few_shot_stance_reason
mistrallargelatest_zero_shot_stance_label
mistrallargelatest_zero_shot_stance_reason
mistrallargelatest_one_shot_stance_label
mistrallargelatest_one_shot_stance_reason
mistrallargelatest_few_shot_stance_label
mistrallargelatest_few_shot_stance_reason
Author_key_name -> corresponding author name of the author key
Stance_id_timestamp -> timestamp of the comment/submission
Stance_id_comment -> textual content of the stance id (per message)
Author_names -> Meta data of authors information for a submission id
Author_id_details -> Metadata of authors comments count and ids of the commen

Majority_vote_stance_label -> Final gold label for stance classification


