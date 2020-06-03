from collections import defaultdict
from typing import List
from random import choices

from utils import trump_tweets


class MarkovTextChain:
    def __init__(self, order: int = 1):
        self.order = order
        self.prob_map = defaultdict(lambda: defaultdict(float)) # TODO: Get rid of this, choices can work with counts
        self._count_map = defaultdict(lambda: defaultdict(int))
    
    def train(self, corpus: List[str]) -> None:
        for text in corpus:
            train_seq = ([None] * self.order) + text.split() + ([None] * self.order)
            cur_state = tuple(train_seq[:self.order])

            for i in range(1, len(train_seq) - self.order):
                next_state = tuple(train_seq[i:i + self.order]) # TODO: incorrect for order > 1
                self._count_map[cur_state][next_state] += 1

                cur_state = next_state
        
        for cur_state, next_state in self._count_map.items():
            denom = sum(next_state.values())
            for state, count in next_state.items():
                self.prob_map[cur_state][state] = count / denom
        
    def generate(self) -> str:
        seq = list()
        cur_state = tuple([None] * self.order)

        while self.prob_map[cur_state].keys():
            cur_state = choices(list(self.prob_map[cur_state].keys()), weights=list(self.prob_map[cur_state].values()))[0]
            seq.append(cur_state[-1])
        
        return " ".join(seq)


if __name__ == '__main__':
    corpus = trump_tweets()['content'].tolist()

    mtc = MarkovTextChain()
    mtc.train(corpus)

    print(mtc.generate())
