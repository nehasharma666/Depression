# **Context is Important in Depressive Language: A Study of the Interaction Between the Sentiments and Linguistic Markers in Reddit Discussions**

**Accepted in:**
14th Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis (WASSA) at ACL 2024.

**Paper Link:**
[Context is Important in Depressive Language: A Study of the Interaction Between the Sentiments and Linguistic Markers in Reddit Discussions](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=zvQvpjIAAAAJ&citation_for_view=zvQvpjIAAAAJ:u-x6o8ySG0sC)

## **Abstract**
Research exploring linguistic markers in individuals with depression has demonstrated that language usage can serve as an indicator of mental health. This study investigates the impact of discussion topic as context on linguistic markers and emotional expression in depression, using a Reddit dataset to explore interaction effects. Contrary to common findings, our sentiment analysis revealed a broader range of emotional intensity in depressed individuals, with both higher negative and positive sentiments than controls. This pattern was driven by posts containing no emotion words, revealing the limitations of the lexicon based approaches in capturing the full emotional context. We observed several interesting results demonstrating the importance of contextual analyses. For instance, the use of 1st person singular pronouns and words related to anger and sadness correlated with increased positive sentiments, whereas a higher rate of present-focused words was associated with more negative sentiments. 
Our findings highlight the importance of discussion contexts while interpreting the language used in depression, revealing that the emotional intensity and meaning of linguistic markers can vary based on the topic of discussion. 

## **Repository Structure**

### 1. **Sentiment and Topic Models and Basic Analysis**
This folder contains the following:

- **Data**: Raw and processed data used in the study.
- **Roberta Sentiment Model**: Code and resources for sentiment analysis using the RoBERTa model.
- **BERTopic Modeling**: Implementation of BERT-based topic modeling.
- **After Analysis**: Post-modeling analyses and results.

### 2. **Topic Specific LIWC Analysis**
This folder includes:

- **LIWC Analysis Topic Wise**: Linguistic Inquiry and Word Count (LIWC) analyses conducted on specific topics identified in the data and Linear regression.
- **Mixed Emotion Analysis**: Examination of mixed emotional expressions.

### 3. **User Based Analysis**
This folder provides:

- **Welch T-Test Analysis**: Statistical analysis comparing depression and control groups using Welch's t-test.

## **How to Use**

1. **Data Preparation**: Navigate to the "Sentiment and Topic Models and Basic Analysis" folder to access and understand the data preprocessing steps.
2. **Sentiment Analysis**: Use the provided RoBERTa model implementation to perform sentiment analysis on the dataset.
3. **Topic Modeling**: Apply the BERTopic modeling techniques available in the folder to extract and analyze topics within the discussions.
4. **LIWC Analysis**: Follow the instructions in the "Topic Specific LIWC Analysis" folder to conduct LIWC analyses on identified topics and examine mixed emotions.
5. **User Analysis**: Utilize the scripts in the "User Based Analysis" folder to compare linguistic markers between depression and control groups using the Welch t-test.

## **Contact**

For any questions or further information, please contact:

- **Neha Sharma**: [neha.sharma@ut.ee](mailto:neha.sharma@ut.ee)
- **Kairit Sirts**: [sirts@ut.ee](mailto:sirts@ut.ee)
