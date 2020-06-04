from collections import defaultdict
from typing import List
from random import choices

from utils import trump_tweets


class MarkovTextChain:
    def __init__(self, order: int = 1):
        self.order = order
        self.state_map = defaultdict(lambda: defaultdict(int))
    
    def train(self, corpus: List[str]) -> None:
        for text in corpus:
            train_seq = ([None] * self.order) + text.split() + ([None] * self.order)

            for i in range(len(train_seq) - self.order):
                cur_state = tuple(train_seq[i:i + self.order])
                next_state = train_seq[i + self.order]
                self.state_map[cur_state][next_state] += 1
        
    def generate(self) -> str:
        seq = list()
        next_word = ""
        cur_state = tuple([None] * self.order) 

        while True:
            next_word = choices(list(self.state_map[cur_state].keys()), weights=list(self.state_map[cur_state].values()))[0]   
            if next_word is None:
                break

            seq.append(next_word)
            cur_state = cur_state[1:] + (next_word,)
        
        return " ".join(seq)


if __name__ == '__main__':
    corpus = trump_tweets()['content'].tolist()

    mtc = MarkovTextChain()
    mtc.train(corpus)

    print(mtc.generate())
