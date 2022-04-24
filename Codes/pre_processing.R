library(tidyverse)
library(tidytext)
library(dplyr)
# read csv file
df <- read.csv('../input/toxiccommentclassification/cleaned_train.csv')
df
# we create c_df --> tibble datatype alongwith adding/ mutating row numbers 
c_df <- data.frame(read.csv('../input/toxiccommentclassification/cleaned_train.csv')) %>% head(20000) %>% as_tibble() %>% mutate(n = row_number())
dim(c_df)
c_df %>% head()
write_csv(c_df, "train.csv")
# unnesting the tokens comment wise
# the arguments for unnesting --> 1. name of column containing newly craeted tokens
# name of column to split
# by default unnest will convert all to lowercase

# 1_gram shinghles
tokens <- c_df %>% unnest_tokens(words, comment_text)
tokens %>% head()
# anti join
cleaned_tokens <- tokens %>%
transmute(word = words, n) %>%
anti_join(stop_words) %>%
as_tibble() %>%
rename(sentence_number = n)


cleaned_tokens <- cleaned_tokens %>%
count(word) %>%
inner_join(cleaned_tokens, by = "word") %>%
arrange(desc(n))
head(cleaned_tokens, 10)

comments_tf_idf <- cleaned_tokens %>% bind_tf_idf(word, sentence_number, n) %>% arrange(desc(n))

# printing the tf-idf values

library(ggplot2)

cleaned_tokens %>%
  count(word, sort = TRUE) %>%
  filter(n > 5200) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(n, word)) +
  geom_col(color = "black", fill = "#6baed6")

# creating 2 character shingles
comments_tf_idf %>% filter(tf_idf < 4 & tf_idf > 0.5) %>% ggplot(aes(x = tf_idf)) + geom_histogram(color = "black", fill = "#6baed6")
ngram_2 <- c_df %>%
unnest_ngrams(words, df.comment_text, n = 2) %>%
rename(sentence_number = n)

ngram_2 <- ngram_2 %>%
count(words) %>%
inner_join(ngram_2) %>%
arrange(desc(n))

ngram_2_tf_idf <- ngram_2 %>% bind_tf_idf(words, sentence_number, n) %>% arrange(desc(tf_idf))
head(ngram_2_tf_idf, 10)

write.csv(ngram_2_tf_idf, "ngram_2_tf_idf.csv")

# 1 character shngles
comment_char <- c_df %>% unnest_characters(word, df.comment_text) %>% rename(sentence_number = n)
head(comment_char, 1)

comment_char <- comment_char %>% count(word) %>% inner_join(comment_char) %>% arrange(desc(n))
comment_char_tf_idf <- comment_char %>% bind_tf_idf(word, sentence_number, n) %>% arrange(desc(tf_idf))

write_csv(comment_char_tf_idf, "comment_char_tf_idf.csv")

# ploting the tf-idf as histogram
library(ggplot2)

d %>% filter(tf_idf < 0.05 & tf_idf > -0.05) %>%
  ggplot(aes(tf_idf)) +
  geom_histogram(color = "black", fill = "#6baed6")