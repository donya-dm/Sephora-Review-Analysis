**Sephora Review Analysis Using NLP**

Google Colab link:
https://colab.research.google.com/drive/1xnvV5VSNTSCCPdVhw9f7-y7rJycZt5bW?authuser=1

Introduction:
This project aims to analyze Sephora customer reviews using NLP techniques to uncover key themes and sentiments in product feedback. The dataset consists of over a million reviews, processed through text cleaning, tokenization, and lemmatization. Exploratory data analysis reveals patterns in review length, sentiment distribution, and product preferences. Topic modeling using LDA and NMF identifies dominant discussion points across skincare, makeup, and fragrance categories. Our findings provide insights into consumer preferences, helping brands improve product offerings and marketing strategies.

Problem Statement:
Businesses in the beauty and skincare industry struggle to understand customer preferences due to fragmented and unstructured feedback. Traditional search filters and product descriptions fail to capture consumer needs, sentiments, and trends, leading to misaligned product offerings and ineffective recommendations. As a result, customers face decision fatigue, poor experiences, and higher return rates. 

Data Set Description:
The dataset for this analysis consists of customer reviews and product descriptions from Sephora, a leading beauty retailer offering a diverse range of skincare, makeup, and fragrance products. The dataset includes user reviews, ratings, product details, price, and review text. Some of the features we use include:
Rating: The average rating of the product based on user reviews
Review Text: The review text written by each user
Product ID: The unique identifier for the product from the site 

Methodology:
This project analyzes Sephora product reviews using five key steps. We first pre-processed the text by cleaning, tokenizing, and removing stopwords. Afterwards in the EDA phase, we examined the dataset structure, review lengths, and word frequency patterns. Sentiment analysis classified reviews into positive, neutral, or negative categories, while topic modeling identified key themes. Finally, we summarized insights to understand customer experiences and trends.

Preprocess/EDA
Data Processing and Text Pre-processing
We processed the data by merging multiple CSV files into a single dataset and selecting a random sample of 10 reviews per product to ensure balance. To handle missing data, we removed rows with missing review_text, ensuring that only complete and meaningful reviews were included in the analysis. This was achieved using the dropna() function, which eliminated rows containing any missing values, thereby preventing potential biases and inconsistencies in the dataset.

To prepare the text for analysis, we applied various pre-processing techniques, including normalization by converting text to lowercase and removing punctuation and digits. We then performed tokenization and stopword removal using NLTK, followed by stemming and lemmatization to standardize word forms. These steps ensured that the data was clean and ready for further analysis.

Exploratory Data Analysis (EDA)
In the EDA phase, we examined the dataset structure, analyzed review length distributions, and identified common word usage patterns. The dataset contains 21,220 reviews, with three key columns: rating, review_text, and product_id. Most reviews are relatively short, with fewer than 100 words, as shown in the distribution analysis. Additionally, the most frequently used words include 'skin,' 'use,' 'product,' 'love,' and 'feel', indicating a strong focus on skincare and customer experience. These analyses provided valuable insights into customer behavior and product perception.

Sentiment Analysis 
To evaluate the performance of products based on customer reviews, we conducted sentiment analysis using three different methods: VADER, TextBlob, and Sentence Transformer. The goal was to identify the best method for capturing sentiment accurately, allowing us to rank products based on customer preferences. The classification for all three methods are based on:
Positive (score > 0.05)
Neutral (-0.05 ≤ score ≤ 0.05)
Negative (score < -0.05)
VADER (Valence Aware Dictionary and sEntiment Reasoner)
Advantage: Fast, rule-based approach optimized for short text, especially social media.
Disadvantage: Lacks contextual understanding and may misinterpret nuanced language.
TextBlob
Advantage: Simple lexicon-based method that assigns polarity scores based on predefined words.
Disadvantage: Struggles with complex sentence structures and negation.
Sentence Transformer
Advantage: Uses deep learning to understand context and sentiment beyond individual words.
Disadvantage: Computationally expensive compared to rule-based methods.
We evaluated the agreement between models to understand how they differ in classification:

                                 Metric      Value
0               Agreement (All 3 Models)  42.370405
1       Agreement (VADER vs Transformer)  49.349670
2          Agreement (VADER vs TextBlob)  81.998115
3    Agreement (Transformer vs TextBlob)  47.657870

Sentiment Distribution Comparison Across Models (Percentage):
          VADER Percentage  Transformer Percentage  TextBlob Percentage
Positive         87.125353               47.266730            81.644675
Negative          7.210179               30.400566             6.559849
Neutral           5.664467               22.332705            11.795476

Based on these results, we selected Sentence Transformer as the most reliable method due to its more balanced sentiment distribution and better contextual understanding. After selecting Sentence Transformer, we ranked products based on their sentiment scores. This allowed us to identify the Top 10 best-rated products and the 10 lowest-rated products based on customer sentiment. These rankings provide valuable insights into customer satisfaction and product performance.

Topic Modeling and Word Cloud
Understanding customer feedback is crucial for brands like Sephora to enhance product offerings and marketing strategies. Given the vast volume of reviews, manually analyzing them is impractical. To address this challenge, we applied Topic Modeling and Word Cloud visualization to extract meaningful insights from customer reviews. These techniques helped identify key themes, recurring phrases, and customer concerns across different product categories.

Topic modeling was used to uncover hidden patterns in customer discussions by categorizing similar words into distinct topics. We implemented three topic modeling techniques: 
Latent Dirichlet Allocation (LDA) using Gensim
LDA using Scikit-learn
Non-Negative Matrix Factorization (NMF)

The Gensim LDA model assigned probabilities to words in different topics, making it effective for document-level analysis. The Scikit-learn LDA model, leveraging TF-IDF (Term Frequency-Inverse Document Frequency), provided clearer topic separations and was ultimately chosen as the best-performing model. Lastly, the NMF approach, which decomposes matrices to identify word co-occurrence patterns, performed well on short text data such as product reviews.

After optimizing the number of topics, we identified six key themes across Sephora reviews. Eye Care products were frequently associated with words like “dark circles,” “puffiness,” and “wrinkles.” Cleansers were described with terms such as “gentle,” “refreshing,” and “makeup remover.” Moisturizers commonly featured words like “hydration,” “smooth,” and “soft.” Reviews discussing product effectiveness included phrases like “notice a difference” and “works great.” Lip care products were often described with words like “lip balm,” “moisture,” and “color.” Lastly, a general sentiment topic emerged, with common words such as “love,” “really great,” and “feel amazing.” These insights provided valuable guidance on what customers emphasize most in their reviews, helping Sephora refine its product offerings and marketing messages.

To supplement topic modeling, we utilized Word Cloud visualizations to represent frequently mentioned words in customer feedback. By removing stopwords and preprocessing the text, we created visual summaries of the most significant words in positive and negative reviews. The Word Cloud for positive reviews prominently featured words like “love,” “hydration,” “amazing,” “soft,” and “glow,” indicating a strong preference for moisturizing and aesthetic benefits. On the other hand, the negative review Word Cloud displayed concerns such as “irritation,” “burn,” “dry,” “smell,” and “expensive,” highlighting potential product drawbacks.

These findings can help Sephora make data-driven decisions, such as adjusting marketing strategies, improving formulations, and addressing frequent customer complaints. By leveraging NLP-based insights, brands can proactively enhance customer satisfaction and optimize their product offerings to better align with consumer expectations.

Summarization:
We also wanted to implement automated summarization of Sephora product reviews to help consumers quickly understand product feedback without reading hundreds of individual reviews. Two approaches were compared (extractive and abstractive summarization): a traditional library like TextRank as extractive summarization and an advanced approach using Google's Gemini API as abstractive summarization.

NLTK with TextRank Based Summarization: Limitations
The initial implementation used NLTK with a TextRank-based extractive summarization approach:
Created sentence vectors and calculated similarity using cosine distance
Applied PageRank algorithm to identify important sentences
Selected top-ranked sentences for the summary

However, the summaries consisted of disconnected words rather than coherent sentences. For example: "moisturizer good skin dry summer perfect use price." While this captured keywords, it failed to provide meaningful, readable information.

Gemini-Based Summarization: Improvements
To address these limitations, we implemented a solution using Google's Gemini API:
Used Gemini to generate abstractive summaries
Prompt engineering to ensure consistent two-sentence summaries
Generated review summaries of each product

The Gemini-based approach generated coherent, contextual summaries in proper sentences. Instead of disconnected words, it created summaries like: "This moisturizer provides excellent hydration for dry skin without feeling greasy. Customers particularly appreciate its light scent, fast absorption, and value for money.

The comparison demonstrates significant advancement in NLP capabilities. While traditional techniques identify statistically important content, they often fail to produce human-readable summaries, especially with large datasets. Gemini's approach leverages advanced language models to understand context and generate coherent summaries that effectively capture the essence of multiple reviews, creating significantly more value for consumers making purchase decisions.

Results and Recommendations:
The review summaries for the top 10 skincare products highlight key strengths that can be used for targeted marketing campaigns. We observed clear patterns where highly-rated products still receive consistent complaints. Few notable examples are the Evian Spray, which we believe should be positioned as a travel essential for hot and dry climates. Additionally, the Origins Rose Clay Mask should be marketed specifically for oily and combination skin to address concerns about dryness. Moreover, we think that some of their products could be reformulated. For instance, the Dior Youth Age-Delay Advanced Crème faces recurring complaints about its strong scent and thick texture. We would suggest reformulating the cream to a fragrance-free, lighter formula. Another example is the Supergoop! Mineral Mattescreen, which should also be reformulated to prevent pilling and reduce the white cast on darker skin tones.

Conclusion:
In summary, this project uses NLP techniques to extract meaningful insights from Sephora reviews, which can be used to create more value for both the company and its customers. Topic modeling identifies key product concerns which enables Sephora to refine formulations, similar to L'Oréal’s AI-driven personalization that boosted conversions by 20%. Sentiment analysis aids inventory decisions, which is a strategy Amazon used to increase profit by 10%. Summarization and word clouds enhance marketing strategies, much like Nike’s data-driven campaigns that improved engagement by 40%. By utilizing these insights, Sephora can increase sales, improve customer satisfaction, and strengthen brand loyalty in a competitive beauty market.
