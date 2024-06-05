# USDC
This repository contains the USDC dataset, and the entire code of **"USDC: A Dataset of User Stance and Dogmatism in Long Conversations"** paper.

# To Download the Dataset
``` bash
pip install data -> just example
```


## USDC Dataset Folder Details
This folder contains the USDC stance and dogmatism datasets. Stance is at post level and given for each comment of the user. Dogmatism is at user level and given for entire user conversation.
There are two sub folders in this folder. The subfolder "USDC Stance Data" contains the "USDC_Stance.pkl" file that contains the information related to the stance classification data and corresponding LLM annotations, the training_stance.pkl file contains the training data and testing_stance.pkl file contains the testing data for Stance classification. 

** The columns in USDC_Stance.pkl file are as follows:**
- Submission_id 
  2. Subreddit ->
  3. title ->
  4. content ->
  5. reddit_link ->
  6. comments -> Nested replies to the submission
  7. author_key -> unique author identification (tells about author1 or author 2)
  8. stance_id -> unique identifier of comment
9. gpt41106preview_zero_shot_stance_label -> Stance annotations generated using GPT-4 zero shot prompting
10. gpt41106preview_zero_shot_stance_reason -> Reasons for selecting the stance label generated using GPT-4 zero shot prompting
11. gpt41106preview_one_shot_stance_label ->
12. gpt41106preview_one_shot_stance_reason ->
13. gpt41106preview_few_shot_stance_label
14. gpt41106preview_few_shot_stance_reason
15. mistrallargelatest_zero_shot_stance_label
16. mistrallargelatest_zero_shot_stance_reason
17. mistrallargelatest_one_shot_stance_label
18. mistrallargelatest_one_shot_stance_reason
19. mistrallargelatest_few_shot_stance_label
20. mistrallargelatest_few_shot_stance_reason
21. author_key_name -> corresponding author name of the author key
22. stance_id_timestamp -> timestamp of the comment/submission
24. stance_id_comment -> textual content of the stance id (per message)
25. author_names -> Meta data of authors information for a submission id
26. author_id_details -> Metadata of authors comments count and ids of the commen
27. majority_vote_stance_label -> Final gold label for stance classification

The "USDC Dogmatism Data" sub folder contains the information related to the dogmatism data, corresponding LLM annotations and the training and testing data for Dogmatism classification.


