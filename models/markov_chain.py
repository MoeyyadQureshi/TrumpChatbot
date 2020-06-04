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
            cur_state = tuple(train_seq[:self.order])

            for i in range(1, len(train_seq) - self.order):
                next_state = tuple(train_seq[i:i + self.order]) # TODO: incorrect for order > 1
                self.state_map[cur_state][next_state] += 1

                cur_state = next_state
        
    def generate(self) -> str:
        seq = list()
        cur_state = tuple([None] * self.order)

        while self.state_map[cur_state].keys():
            cur_state = choices(list(self.state_map[cur_state].keys()), weights=list(self.state_map[cur_state].values()))[0]
            seq.append(cur_state[-1])
        
        return " ".join(seq)


if __name__ == '__main__':
    corpus = trump_tweets()['content'].tolist()

    mtc = MarkovTextChain()
    mtc.train(corpus)

    print(mtc.generate())
