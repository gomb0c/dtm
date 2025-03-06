import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from nltk import TreePrettyPrinter, Tree

from TPR_utils import BatchSymbols2NodeTree, DecodedTPR2Tree, build_D, build_E
import torch.utils.checkpoint as checkpoint
from vector_symbolic_utils import VectorSymbolicConverter, VectorSymbolicManipulator


class DiffTreeMachine(nn.Module):
    def __init__(self, d_filler, d_role, d_model, role_emb, steps, dim_feedforward, nhead=4, dropout=.1,
                 transformer_activation='gelu', layer_norm_eps=1e-5, transformer_norm_first=True,
                 transformer_layers_per_step=1, op_dist_fn='softmax', arg_dist_fn='softmax', ind2vocab=None, 
                 vector_symbolic_converter: VectorSymbolicConverter=None, vector_symbolic_manipulator: VectorSymbolicManipulator=None):
        super().__init__()
        d_tpr = d_filler * d_role

        self.ctrl_net = nn.Linear(d_tpr, d_model)
        self.num_ops=3

        self.interpreter = DiffTreeInterpreter(vector_symbolic_manipulator)

        self.steps = steps

        self.nta = NeuralTreeAgent(steps, d_model, nhead, dim_feedforward, dropout, d_filler, self.num_ops,
                                   transformer_activation, layer_norm_eps, transformer_norm_first,
                                   transformer_layers_per_step, op_dist_fn, arg_dist_fn)
        self.op_logits_token = nn.parameter.Parameter(torch.Tensor(1, d_model))
        nn.init.normal_(self.op_logits_token)
        self.root_filler_token = nn.parameter.Parameter(torch.Tensor(1, d_model))
        nn.init.normal_(self.root_filler_token)

        # ind2vocab will be used for debugging in forward()
        self.ind2vocab = ind2vocab
        self.vector_symbolic_converter = vector_symbolic_converter

    def forward(self, input_tpr, debug=False, calculate_entropy=False):
        debug_writer = [] if debug else None
        # We need to convert indices to words for printing
        if debug:
            assert self.ind2vocab

        bsz = input_tpr.shape[0]

        memory = input_tpr.unsqueeze(1) # the steps dimension

        # Setup the encodings for the NTA
        op_logits_token = self.op_logits_token.repeat(bsz, 1, 1)
        root_filler_token = self.root_filler_token.repeat(bsz, 1, 1)

        encodings = torch.cat((op_logits_token, root_filler_token), dim=1)

        weight_dict = {} 
        for step in range(self.steps):
            # Encode the most recent TPR in memory

            tree_to_shrink = memory[:, step]
            tree_encoding = self.ctrl_net(tree_to_shrink.flatten(1))

            encodings = torch.cat((encodings, tree_encoding.unsqueeze(1)), dim=1)

            op_dist, root_filler, arg_weights, encodings = self.nta(encodings, step)

            new_tree = self.interpreter(memory[:, :step+1], arg_weights, root_filler, op_dist)

            if debug:
                output_string = 'Layer {}:\nMemory:'.format(step)
                debug_writer.append(output_string)
                # Use the batch dimension to decode previous layers in memory
                x_decoded = DecodedTPR2Tree(self.vector_symbolic_converter.decode_vs_to_tree(memory[0], True))
                x_tree = BatchSymbols2NodeTree(x_decoded, self.ind2vocab)
                for tree in x_tree:
                    if tree:
                        debug_writer.append(tree.str())
                debug_writer.append('car: {:.3f}\tcdr: {:.3f}\tcons: {:.3f}'.format(op_dist[0][0], op_dist[0][1], op_dist[0][2]))
                debug_writer.append('car argument weight: {}'.format(
                    np.array2string(arg_weights[0,:,0].detach().cpu().numpy(), precision=2)))
                debug_writer.append('cdr argument weight: {}'.format(
                    np.array2string(arg_weights[0, :, 1].detach().cpu().numpy(), precision=2)))
                debug_writer.append('cons1 argument weight: {}'.format(
                    np.array2string(arg_weights[0, :, 2].detach().cpu().numpy(), precision=2)))
                debug_writer.append('cons2 argument weight: {}'.format(
                    np.array2string(arg_weights[0, :, 3].detach().cpu().numpy(), precision=2)))
                weight_dict[step] = {'op_dist': op_dist[0],
                                     'tree_dist_car': arg_weights[0,:,0],
                                     'tree_dist_cdr': arg_weights[0,:,1],
                                     'tree_dist_cons1': arg_weights[0,:,2],
                                     'tree_dist_cons2': arg_weights[0,:,3]}
                weight_dict[step] = dict(
                    map(lambda kv: (kv[0], kv[1].detach().cpu().numpy()),
                        weight_dict[step].items())
                )
                
                fully_decoded = DecodedTPR2Tree(
                    self.vector_symbolic_converter.decode_vs_to_tree(new_tree[0].unsqueeze(0), True))
                debug_tree = BatchSymbols2NodeTree(fully_decoded, self.ind2vocab)[0]
                debug_writer.append('Output: ')
                if not debug_tree:
                    debug_writer.append('None')
                else:
                    pretty_tree = TreePrettyPrinter(Tree.fromstring(debug_tree.str()))
                    debug_writer.append('```{}```'.format(pretty_tree.text()))

            memory = torch.cat([memory, new_tree.unsqueeze(1)], dim=1)

        debug_info = None
        if debug:
            #print('\n'.join(debug_writer))
            debug_info = {'text': debug_writer,
                          'weight_dict': weight_dict}

        return memory[:, -1], debug_info, None

    def set_gumbel_temp(self, temp):
        self.interpreter.gumbel_temp = temp
        self.nta.gumbel_temp = temp


class NeuralTreeAgent(nn.Module):
    '''
    The Neural Tree Agent
    '''
    def __init__(self, steps, d_model, nhead, dim_feedforward, dropout, d_filler, num_ops, activation, layer_norm_eps,
                 transformer_norm_first, transformer_layers_per_step, op_dist_fn, arg_dist_fn):
        super().__init__()
        # We only need to create a single layer since this layer will be deep copied by nn.TransformerEncoder
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                       dropout=dropout, activation=activation,
                                                       layer_norm_eps=layer_norm_eps,
                                                       batch_first=True, norm_first=bool(transformer_norm_first))

        self.op_dist_fn = op_dist_fn
        self.arg_dist_fn = arg_dist_fn

        self.layers = nn.ModuleList()
        self.arg_logits_list = nn.ModuleList()
        self.root_filler_list = nn.ModuleList()
        self.op_logits_list = nn.ModuleList()
        for i in range(steps):
            encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if transformer_norm_first else None
            self.layers.append(nn.TransformerEncoder(transformer_layer, transformer_layers_per_step, encoder_norm))

            # 4 for the 4 arguments, car, cdr, cons1, cons2
            arg_logits = nn.Linear(d_model, 4)
            self.arg_logits_list.append(arg_logits)

            root_filler = nn.Linear(d_model, d_filler)
            self.root_filler_list.append(root_filler)

            op_logits = nn.Linear(d_model, num_ops)
            self.op_logits_list.append(op_logits)

    def forward(self, encodings, step):
        # TODO: move pashamax, sparsemax, softmax from the old car/cdr/consnet to here for op_dist and arg_weights
        encodings = self.layers[step](encodings)

        op_logits = self.op_logits_list[step](encodings[:, 0, :])
        if self.op_dist_fn == 'softmax':
            op_dist = F.softmax(op_logits, dim=-1)
        elif self.op_dist_fn == 'gumbel':
            op_dist = F.gumbel_softmax(op_logits, tau=self.gumbel_temp)
        else:
            raise ValueError('Unknown op_dist_fn: {}'.format(self.op_dist_fn))
        root_filler = self.root_filler_list[step](encodings[:, 1, :])

        arg_logits = self.arg_logits_list[step](encodings[:, 2:, :])
        if self.arg_dist_fn == 'softmax':
            arg_weights = F.softmax(arg_logits, dim=1)
        elif self.arg_dist_fn == 'gumbel':
            arg_weights = F.gumbel_softmax(arg_logits, tau=self.gumbel_temp)
        else:
            raise ValueError('Unknown arg_dist_fn: {}'.format(self.arg_dist_fn))

        return op_dist, root_filler, arg_weights, encodings


class DiffTreeInterpreter(nn.Module):
    def __init__(self, vs_manipulator: VectorSymbolicManipulator):
        super().__init__()
        self.vs_manipulator = vs_manipulator

        

    def forward(self, memory, arg_weights, root_filler, op_dist, calculate_entropy=False):
        memory_shape = list(memory.shape)
        # The length index should be changed into the num_ops index
        memory_shape[1] = self.num_ops
        full_output = torch.empty(memory_shape, device=memory.device)

        car_arg_weights = arg_weights[:, :, 0]
        cdr_arg_weights = arg_weights[:, :, 1]
        cons_arg1_weights = arg_weights[:, :, 2]
        cons_arg2_weights = arg_weights[:, :, 3]

        full_output[:, 0] = self.vs_manipulator.apply_car(memory,
                                         arg1_weight=car_arg_weights)
        full_output[:, 1] = self.vs_manipulator.apply_cdr(memory,
                                         arg1_weight=cdr_arg_weights)

        # Each of these functions has a large memory usage for calculating the blended argument
        full_output[:, 2] = self.vs_manipulator.apply_cons(memory, arg1_weight=cons_arg1_weights, arg2_weight=cons_arg2_weights,
                                          root_filler=root_filler)

        return torch.einsum('bnfr,bn->bfr', full_output, op_dist)

