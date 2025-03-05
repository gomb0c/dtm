import random
import torch
from torch.utils.data import Dataset
import json

from node import text_tree_to_node

'''
Example: 
Vocab2ind is {'<empty>': 0, 'RCONS': 1, 'DET': 2, 'one': 3, 
'NOUN': 4, 'rabbit': 5, 'R': 6, 'some': 7, 'cactus': 8,
'CDR': 9, 'her': 10, 'weed': 11, 'his': 12, 'CAR': 13, 
'my': 14, 'bear': 15, 'fox': 16, 'sheep': 17, 'the': 18, 
'reindeer': 19, 'burro': 20, 'pig': 21, 'your': 22, 'flower': 23, 
'our': 24, 'donkey': 25, 'cat': 26, 'a': 27, 'cheetah': 28, 'crocodile': 29, 
'goat': 30, 'snake': 31, 'horse': 32, 'rose': 33, 'dog': 34, 'leopard': 35, 
'bamboo': 36, 'tree': 37, 'giraffe': 38, 'cow': 39, 'blossom': 40, 
'zebra': 41, 'lamb': 42, 'elephant': 43}

Sample is ( CDR ( DET her ) ( NOUN bamboo ) ), 
transformed is tensor([9, 2, 4,  ..., 0, 0, 0], device='cuda:0')
'''

class BinaryT2TDataset(Dataset):
    '''
    Trees are represented as vectors of indices of length 2**depth
    '''

    def __init__(self, jsonl_file, max_examples=None, max_depth=20, ind2vocab=None, vocab2ind=None, device='cuda'):
        self.device = device
        self.max_depth = max_depth
        if ind2vocab is None:
            self.ind2vocab = ['<empty>']
            self.vocab2ind = {'<empty>': 0}
        else:
            self.ind2vocab = ind2vocab
            self.vocab2ind = vocab2ind

        with open(jsonl_file) as f:
            rows = list(f)

            if max_examples:
                random.shuffle(rows)
                rows = rows[0:max_examples]

            self.data = self.process_trees(rows)

        print(f'Vocab2ind is {vocab2ind}\n')

    def process_trees(self, data):
        processed = []
        max_branch = 0

        dataset_max_depth = 0

        for line in data:
            inout_pair = json.loads(line)
            in_node = text_tree_to_node(inout_pair['source'])
            out_node = text_tree_to_node(inout_pair['target'])

            example = {"input": in_node, "output": out_node, "example_type": None}

            max_branch = max([max_branch, in_node.get_max_branching(), out_node.get_max_branching()])
            assert max_branch <= 2

            if example['input'].get_max_depth() > self.max_depth or example['output'].get_max_depth() > self.max_depth:
                continue
            
            # add to vocab
            def _add_to_vocab(node):
                if node is None:
                    return
                if node.label not in self.vocab2ind:
                    self.ind2vocab.append(node.label)
                    self.vocab2ind[node.label] = len(self.ind2vocab) - 1
                for child in node.children:
                    _add_to_vocab(child)
                return
            
            _add_to_vocab(example['input'])
            _add_to_vocab(example['output'])

            processed.append(example)

        return processed

    def __getitem__(self, idx):
        '''
        Gets the specified text input/output and converts it to a set of tensors.
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get input/output node trees as dict
        sample = self.data[idx]

        transformed_sample = {}
        transformed_sample['example_type'] = sample['example_type'] if sample['example_type'] is not None else 0
        transformed_sample['input'] = self.text_to_tensors(sample['input'])
        transformed_sample['output'] = self.text_to_tensors(sample['output'])
        
        input = sample['input']
        transformed = transformed_sample['input']
        print(f'Sample is {input}, transformed is {transformed}')
        return transformed_sample
    
    def text_to_tensors(self, node):
        sample_tensor = torch.zeros((2**self.max_depth-1, ), device=self.device, dtype=torch.long)
        def _traverse_and_tensorify(node, ind):
            if node is None:
                return
            sample_tensor[ind] = self.vocab2ind[node.label]
            if len(node.children) > 0:
                # work on the left child
                _traverse_and_tensorify(node.children[0], ind*2+1)
            if len(node.children) > 1:
                # work on the right child
                _traverse_and_tensorify(node.children[1], ind*2+2)
            return
        _traverse_and_tensorify(node, 0)
        return sample_tensor

    def __len__(self):
        return len(self.data)
