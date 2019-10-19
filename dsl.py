import numpy as np


### Methods
def out_of_bounds(r, c, shape):
    return (r < 0 or c < 0 or r >= shape[0] or c >= shape[1])

def condition(local_program, cell, obs):
    return local_program(cell, obs)

def shifted(direction, local_program, cell, obs):
    if cell is None:
        new_cell = None
    else:
        new_cell = (cell[0] + direction[0], cell[1] + direction[1])
    return local_program(new_cell, obs)

def cell_is_value(value, cell, obs):
    if cell is None or out_of_bounds(cell[0], cell[1], obs.shape):
        focus = None
    else:
        focus = obs[cell[0], cell[1]]

    return (focus == value)

def at_cell_with_value(value, local_program, obs):
    matches = np.argwhere(obs == value)
    if len(matches) == 0:
        cell = None
    else:
        cell = matches[0]
    return local_program(cell, obs)

def at_action_cell(local_program, cell, obs):
    return local_program(cell, obs)

def test_program():
    return False

def scanning(direction, true_condition, false_condition, cell, obs, max_timeout=50):
    if cell is None:
        return False

    for _ in range(max_timeout):
        cell = (cell[0] + direction[0], cell[1] + direction[1])

        if true_condition(cell, obs):
            return True

        if false_condition(cell, obs):
            return False

        # prevent infinite loops
        if out_of_bounds(cell[0], cell[1], obs.shape):
            return False

    return False



### Grammatical Prior
START, CONDITION, LOCAL_PROGRAM, DIRECTION, POSITIVE_NUM, NEGATIVE_NUM, VALUE = range(7)

def get_grammar_regex(object_types):
    pos_int_regex = '[1-9]\d*'
    neg_int_regex = '-' + pos_int_regex
    mod_pos_regex = '[2-9]\d*'
    mod_neg_regex = '-' + mod_pos_regex

    regex = {
        START : ('at_cell_with_value', 'at_action_cell', 'test_program'),
        LOCAL_PROGRAM : ('condition', 'shifted'),
        CONDITION : ('cell_is_value', 'scanning'),
        DIRECTION : ('( ' + pos_int_regex + ' , 0)', '(0, ' + pos_int_regex + ' )',
                     '( ' + neg_int_regex + ' , 0)', '(0, ' + neg_int_regex + ' )',
                     '( ' + pos_int_regex + ' , ' + pos_int_regex + ' )', '( ' + neg_int_regex + ' , ' + pos_int_regex + ' )',
                     '( ' + pos_int_regex + ' , ' + neg_int_regex + ' )', '( ' + neg_int_regex + ' , ' + neg_int_regex + ' )'),
        POSITIVE_NUM : (' 1 ',  ' ' + mod_pos_regex + ' '),
        NEGATIVE_NUM : (' -1 ', ' ' + mod_neg_regex + ' '),
        VALUE : tuple(object_types)
    }
    return regex

def create_grammar(object_types, feature_probs):
    grammar_regex = get_grammar_regex(object_types)
    grammar = {
        START : ([['at_cell_with_value(', VALUE, ',', LOCAL_PROGRAM, ', s)'],
                  ['at_action_cell(', LOCAL_PROGRAM, ', a, s)'],
                  ['test_program()']],
                  [feature_probs[regex] for regex in grammar_regex[START]]),
        LOCAL_PROGRAM : ([['lambda cell,o : condition(', CONDITION, ', cell, o)'],
                          ['lambda cell,o : shifted(', DIRECTION, ',', CONDITION, ', cell, o)']],
                          [feature_probs[regex] for regex in grammar_regex[LOCAL_PROGRAM]]),
        CONDITION : ([['lambda cell,o : cell_is_value(', VALUE, ', cell, o)'],
                      ['lambda cell,o : scanning(', DIRECTION, ',', LOCAL_PROGRAM, ',', LOCAL_PROGRAM, ', cell, o)']],
                      [feature_probs[regex] for regex in grammar_regex[CONDITION]]),
        DIRECTION : ([['(', POSITIVE_NUM, ', 0)'], ['(0,', POSITIVE_NUM, ')'],
                      ['(', NEGATIVE_NUM, ', 0)'], ['(0,', NEGATIVE_NUM, ')'],
                      ['(', POSITIVE_NUM, ',', POSITIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', POSITIVE_NUM, ')'],
                      ['(', POSITIVE_NUM, ',', NEGATIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', NEGATIVE_NUM, ')']],
                     [feature_probs[regex] for regex in grammar_regex[DIRECTION]]),
        POSITIVE_NUM : ([['1'], [POSITIVE_NUM, '+1']],
                         [feature_probs[regex] for regex in grammar_regex[POSITIVE_NUM]]),
        NEGATIVE_NUM : ([['-1'], [NEGATIVE_NUM, '-1']],
                         [feature_probs[regex] for regex in grammar_regex[NEGATIVE_NUM]]),
        VALUE : (object_types, 
                 [feature_probs[regex] for regex in grammar_regex[VALUE]])
    }
    return grammar
    
