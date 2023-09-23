# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    
    robots_url = "https://www.gutenberg.org/robots.txt"
    robots_response = requests.get(robots_url)
    if "User-agent: *" in robots_response.text and "Disallow: /" in robots_response.text:
        print("Pausing to comply with robots.txt policy...")
        time.sleep(10) 

    response = requests.get(url)
    book_content = response.text

    start_comment = "*** START OF THIS PROJECT GUTENBERG EBOOK BEOWULF ***"
    end_comment = "*** END OF THE PROJECT GUTENBERG EBOOK BEOWULF ***"
    start_index = book_content.find(start_comment) + len(start_comment)
    end_index = book_content.find(end_comment)

    contents = book_content[start_index:end_index]

    contents = contents.replace("\r\n", "\n")

    return contents


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    book_string = re.sub(r'\n{2,}', '\n', book_string)
    
    tokens = re.findall(r'\w+|[^\w\s]', book_string)
    
    
    tokens.insert(0, '\x02')
    tokens.append('\x03')
    
    return tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):

    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):

        unique_tokens = set(tokens)
        total_tokens = len(unique_tokens)
        probabilities = pd.Series(1 / total_tokens, index=unique_tokens)
        return probabilities
    
    def probability(self, words):

        return self.mdl.loc[list(words)].prod() if all(word in self.mdl.index for word in words) else 0
        
    def sample(self, M):

        sampled_tokens = self.mdl.sample(n=M, replace=True).index.tolist()
        return ' '.join(sampled_tokens)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):

        self.mdl = self.train(tokens)
    
    def train(self, tokens):

        token_counts = pd.Series(tokens).value_counts()
        total_tokens = len(tokens)
        probabilities = token_counts / total_tokens
        return probabilities
    
    def probability(self, words):

        return self.mdl.loc[words].prod() if words in self.mdl.index else 0
        
    def sample(self, M):

        sampled_tokens = self.mdl.sample(n=M, replace=True, weights=self.mdl.values).index.tolist()
        return ' '.join(sampled_tokens)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        ngrams = []
        for i in range(len(tokens) - self.N + 1):
            ngram = tuple(tokens[i:i + self.N])
            ngrams.append(ngram)
        return ngrams
    
        
    def train(self, ngrams):

        # # N-Gram counts C(w_1, ..., w_n)
        ngram_counts = {}
        for ngram in ngrams:
            if ngram in ngram_counts:
                ngram_counts[ngram] += 1
            else:
                ngram_counts[ngram] = 1

        # # (N-1)-Gram counts C(w_1, ..., w_(n-1))
        n1gram_counts = {}
        for ngram in ngrams:
            n1gram = ngram[:-1]
            if n1gram in n1gram_counts:
                n1gram_counts[n1gram] += 1
            else:
                n1gram_counts[n1gram] = 1

        # # Create the conditional probabilities
        probabilities = []
        for ngram, count in ngram_counts.items():
            n1gram = ngram[:-1]
            n1gram_count = n1gram_counts.get(n1gram, 0)
            prob = count / n1gram_count
            probabilities.append((ngram, n1gram, prob))

        # # Put it all together
        model = pd.DataFrame(probabilities, columns=['ngram', 'n1gram', 'prob'])
        return model

    
    def probability(self, words):

        probability = 1.0
        count = 0
        index = 0
        curr_mdl = None
        curr = self
        
        for i in range(len(words)):
            if i - (self.N - 1) < 0:
                current_gram = tuple(words[0 : i + 1])
            else:
                current_gram = tuple(words[i - (self.N - 1) : i + 1])

            tup_size = len(current_gram)

            if curr.N == tup_size:
                curr_mdl = curr.mdl
            else:
                while curr.N > tup_size:
                    if curr.N == 2:
                        curr_mdl = curr.prev_mdl.mdl
                        break
                    else:
                        curr = curr.prev_mdl
            if tup_size == 1:
                if current_gram[0] not in curr_mdl.index:
                    return 0.0
                probability = curr_mdl[current_gram[0]] * probability
            else:
                for i, gram in enumerate(curr_mdl["ngram"]):
                    if current_gram == gram:
                        count += 1
                        index = i
                        break
                if count == 0:
                    return 0.0
                probability = curr_mdl.iloc[index]["prob"] * probability

        return probability
    
    

    def sample(self, M):

        def recursive_sample(tokens, num_tokens):
            if num_tokens == M:
                return tokens
            prev_ngram = tuple(tokens[-self.N + 1:])
            if prev_ngram in self.mdl['ngram'].values:
                filtered_mdl = self.mdl.loc[self.mdl['ngram'].apply(lambda x: x[:-1] == prev_ngram)]
                next_token = np.random.choices(filtered_mdl['ngram'].apply(lambda x: x[-1]), filtered_mdl['prob'])[0]
            else:
                next_token = '\x03' 

            tokens.append(next_token)
            return recursive_sample(tokens, num_tokens + 1)

        tokens = ['\x02']  
        tokens = recursive_sample(tokens, 1)
        tokens.append('\x03')  
        return ' '.join(tokens)

        

    
