
### Introduction
**Objective**: Analyze Reddit conversations to identify the stance of specific authors on sociopolitical topics and determine their level of dogmatism.
**Stance Defintion**: Stance is defined as the expression of the author's standpoint and judgement towards a given topic.
**Dogmatism Defintion**:  Dogmatism is an opinion strongly believed as a fact to support a stance without a question or allowance for conversation.
**Task**: Given a JSON formatted Reddit post and its comment thread, classify the stance of text segments related to "author1" and "author2" by assigning one of the following five predefined stance labels: `strongly_against`, `somewhat_against`, `somewhat_in_favor`, `strongly_in_favor`, `stance_not_inferrable`. Also, assign a dogmatism label for each author by assigning one of the following four predefined labels: `Deeply Rooted`, `Firm but Open`, `Open to Dialogue`, `Flexible`

### Stance Labels Description:
1. **strongly_aganist / strongly_in_favor**: Mark text showing strong opinions, emotional expressions, or argumentative tones.
2. **somewhat_against / somewhat_in_favor**: Identify texts with openness to discussion, less certainty, or showing interest in different viewpoints.
3. **stance_not_inferrable**: Use for texts that are neutral, support both stances, or where the stance is unclear despite being on-topic.

### Dogmatism Labels Description:
1. **Deeply Rooted**: Reflects a strong, unchangeable belief. This label conveys the idea of someone who is firm in their opinion and unlikely to be swayed.
2. **Firm but Open**: Indicates a person who is not likely to change their mind but does not impose their views authoritatively. It captures the essence of being steadfast in one’s beliefs without being dismissive of others.
3. **Open to Dialogue**: Describes someone who holds a certain opinion but is genuinely interested in considering other viewpoints. This label suggests a willingness to engage in meaningful conversation about differing perspectives.
4. **Flexible**: Denotes a person who is not firmly committed to their stance and is open to changing their opinion. This label is indicative of flexibility and openness to new information or arguments.

### Input Data Format
The input data will be in JSON format and should include several key elements to represent a Reddit submission and its associated comments. Each element provides specific information as described below:

- `id`: This is the unique identifier for the Reddit submission.
- `title`: The title of the post. This is what users see first and often summarizes or hints at the content of the submission.
- `content`: The main post's detailed description. This text segment provides the core message or information the author wishes to communicate with the Reddit community. It may include narratives, questions, or any information relevant to the title.
- `author1` or `author2`: The username of our focus author. This field is applicable if the post or comment is made by one of the specific authors we are tracking in the dataset.
- `comments`: An array (list) of comments related to the Reddit submission. Each comment in this array includes the following fields:
    - `id`: The unique identifier for the comment, allowing for identification and reference within the dataset.
    - `author1` or `author2`: The username of the comment's author, if it is made by one of our focus authors. This helps in tracking contributions by specific individuals.
    - `body`: The text of the comment. This is the main content of the comment where the author responds to the post or another comment, providing insights, opinions, or further information.
    - `replies`: An array of comments that are direct responses to this comment. The structure of each reply follows the same format as the initial comment, including `id`, `author1` or `author2` (if applicable), `body`, and potentially more `replies`.

### Output Data Format
Submit your annotations in JSON format, grouping all stance annotations under the key "stance_annotations". Each entry should be a dictionary containing the segment's "id", your "label", and the "reason" for your choice. Include the dogmatism label and its justification under "dogmatism_label" and "dogmatism_reason", respectively.

The output should follow this structure:
```json
{
  "author1": {
    "name": "[author_name]",
    "stance_annotations": [
      {
        "id": "[segment_id]",
        "label": "[chosen_label]",
        "reason": "[Justification in <50 words]"
      },
      ...
    ],
    "dogmatism_label": "[chosen_dogmatism_label]",
    "dogmatism_reason": "[Justification in <50 words]"
  },
  "author2": {
    "name": "[author_name]",
    "stance_annotations": [
      {
        "id": "[segment_id]",
        "label": "[chosen_label]",
        "reason": "[Justification in <50 words]"
      },
      ...
    ],
    "dogmatism_label": "[chosen_dogmatism_label]",
    "dogmatism_reason": "[Justification in <50 words]"
  }
}
```

### Instructions for Effective Annotation

1. **Labeling Stance**: For each segment (including the original Reddit submission, comments, or replies) where "author1" or "author2" is mentioned, assign a stance label that best represents the stance expressed towards the discussed topic in the conversation. This comprehensive approach ensures no relevant contribution by "author1" or "author2" is overlooked. Evaluate the stance based on the content's tone, argumentation, and engagement level with the topic.
2. **Providing Justification**: For each label assigned, include a concise reason, aiming for less than 50 words. Focus on the stance and argumentative indicators present in the text. 
3. **Dogmatism Assessment**: After reviewing all segments from "author1" and "author2", assign a single dogmatism label reflecting the overall tone and approach in their contributions.

### Additional Considerations

- **Contextual Analysis**: Always consider the broader context provided by preceding text in the thread.
- **Objectivity**: Remain unbiased. Focus solely on the textual evidence and these guidelines.
