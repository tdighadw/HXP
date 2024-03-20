### Generate the length-k scenarios ###
from copy import deepcopy
import sys

YELLOW = 1
RED = -1
EMPTY = 0
TOKENS =[YELLOW, RED]

#  Get from a state-action couple, the entire/part of the transitions available, i.e. the new states associated with
#  their probabilities
#  Input: agent's state (int list list), action (int), environment (Connect4), importance score method (str), number of
#  exhaustive/deterministic steps (int), additional information (dictionary), importance type (str)
#  Output: list of transition-probability couples (couple list)
def transition(s, a, env, approx_mode, exh_steps=0, det_tr=0, add_info=None, imp_type=None):
    s = deepcopy(s)
    opponent = add_info['opponent']
    net = add_info['net']
    # Temporary state
    s = env.update_state(s, opponent.token * (-1), a)
    #  Avoid useless opponent transitions when agent already won
    if env.win(opponent.token * (-1), copy_board=s) and imp_type == 'action':
        return [(1.0, s)]

    transitions = []
    # Look all possible transitions from s
    if approx_mode == 'none' or exh_steps:
        actions = get_actions(s, env)
        for t in actions:
            new_s = deepcopy(s)
            new_s = env.update_state(new_s, opponent.token, t)
            transitions.append((1/len(actions), new_s))
        return transitions

    else:
        # Look for the most probable transition
        if approx_mode == 'last':
            return extract_transitions(1, s, env, opponent, net)
        # Select the 'det' most probable transition(s)
        else:
            return extract_transitions(det_tr, s, env, opponent, net)

#  Check whether the state is terminal or not
#  Input: state (int list list), environment (Connect4), additional information (dict)
#  Output: (bool)
def terminal(s, env, add_info):
    # win/lose/draw conditions
    return env.win(YELLOW, copy_board=s, hxp=True) or not sum(line.count(0) for line in s)

#  Extract n most probable transitions
#  Input: number of transition to extract (int), state (int list list), environment (Connect4), opponent player (Agent),
#  NN (DQN)
#  Output: most probable transition(s) (tuple list)
def extract_transitions(n, s, env, opponent, net):
    if n == 1:
        opp_action = opponent.predict(env.inverse_board(s), net)
        return [(1.0, env.update_state(s, opponent.token, opp_action))]

    #  n most probable transitions, i.e. the most valuable opponent's actions (based on Q values)
    else:
        actions = best_actions(n, env.inverse_board(s), env, opponent, net)
        transitions = []
        for a in actions:
            transitions.append((1 / len(actions), env.update_state(deepcopy(s), opponent.token, a)))

        return transitions

#  Get the n most valuable action of the opponent from a specific state
#  Input: number of actions (int), state (int list list), environment (Connect4), opponent player (Agent), NN (DQN)
#  Output: actions list (int list)
def best_actions(n, s, env, opponent, net):
    #  Illegal actions
    illegal_actions = [i for i in range(env.action_space.n) if s[0][i] != 0]
    #  Sort index based on q values from max to min
    max_idx_list = opponent.sort_actions(s, net)
    # give the n most valuable actions according to Q values
    actions = []
    for a in max_idx_list:
        if a not in illegal_actions and len(actions) < n:
            actions.append(a)  # equiprobable transitions

    return actions

### Compute importance score ###

#  Multiple-tasks function to update some data during the HXP process
#  Input: environment (Connect4), agent (Agent), location of the modification in the HXP process (str),
#  state-action list (int list list-int list), additional information (dictionary)
#  Output: variable
def preprocess(env, agent, location, s_a_list=None, add_info=None):
    if location == 'copies':
        add_info['opponent'].random = False
        env_copy = deepcopy(env)
        return env_copy, agent

    elif location == 'hxp':
        env.board = s_a_list[0]
        return env, agent

    #  Flatten the 2D list into 1D list
    elif location == 'pre_locally_minimal_paxp':
        add_info['lose_weak_paxp'] = True
        return to_flattened_state(s_a_list[0])

    #  From 1D list to 2D list
    elif location == 'post_locally_minimal_paxp':
        del add_info['lose_weak_paxp']
        return to_state(s_a_list[0], env)

    elif location == 'weak_paxp':
        return env, agent, s_a_list

    else:
        return env, agent

#  Check whether an importance score can be computed or not
#  Input: action (int), importance type (str), additional information (dictionary)
#  Output: (bool)
def constraint(action, imp_type, add_info):
    return False

#  Sample n valid states
#  Input: environment (Connect4), partial state(int list list), index of feature to remove from v (int), number of
#  samples to generate (int), additional information (dict)
#  Output: list of states (int list list list)
def sample(env, v, i, n, add_info=None):
    #print('v[i] before sampling: {}'.format(v[i]))
    tmp_value = v[i]
    v[i] = None
    #print('v[i] for sampling: {}'.format(v[i]))
    feature_values = [EMPTY, YELLOW, RED]

    # Get the number of set YELLOW and RED tokens
    # Extract a dict column: [feature_vals, feature_indexes, column, nb] for each column where at least a None feature is
    r_tokens_set = 0
    y_tokens_set = 0
    none_features_nb = 0
    none_dict = {c: [deepcopy(feature_values), [], [], 0] for c in range(env.cols)}
    # dict only used for removing feature values (warning: some generated columns are still 'impossible' after this step)

    token_top = {c: 0 for c in range(env.cols)}
    empty_bottom = {c: 0 for c in range(env.cols)}
    column = 0

    for idx in range(len(v)):
        j = idx % env.rows if idx else 0
        feature = v[j * env.cols + column]

        #  Generate column
        none_dict[column][2].append(feature)
        #  Updates
        if feature is None:
            none_dict[column][1].append(i)
            none_dict[column][3] += 1
            none_features_nb += 1
            # reset cpt
            if empty_bottom[column]:
                empty_bottom[column] = 0

        elif feature is YELLOW:
            y_tokens_set += 1
            if not none_dict[column][3]:
                token_top[column] += 1

        elif feature is RED:
            r_tokens_set += 1
            if not none_dict[column][3]:
                token_top[column] += 1

        elif feature is EMPTY and none_dict[column][3]:
            empty_bottom[column] += 1

        # Change column and remove useless feature values in case of None feature membership to the column
        if j == env.rows - 1:
            if none_dict[column][3]:
                # remove empty value
                if token_top[column]:
                    #print('remove empty value: column: {}'.format(none_dict[column][2]))
                    none_dict[column][0].remove(EMPTY)

                # remove token values
                elif empty_bottom[column]:
                    none_dict[column][0].remove(YELLOW)
                    none_dict[column][0].remove(RED)
                    #print('remove token values: column: {}'.format(none_dict[column][2]))

            column += 1
    #print('v: {}'.format(v))
    #print('r_tokens_set: {} - y_tokens_set: {} - none_features_nb: {}'.format(r_tokens_set, y_tokens_set, none_features_nb))
    #print('none_dict: {} \n token_top: {} \n empty_bottom: {}'.format(none_dict, token_top, empty_bottom))
    #print('none_features_nb', none_features_nb)

    # Get max token numbers of each player on the partial board
    if y_tokens_set == r_tokens_set:
        max_tokens = y_tokens_set + int(none_features_nb//2)

    else:
        max_y_r_nb = max(y_tokens_set, r_tokens_set)
        tmp_nb = none_features_nb - (max_y_r_nb - min(y_tokens_set, r_tokens_set))
        max_tokens = max_y_r_nb if not tmp_nb else max_y_r_nb + int(none_features_nb//2)

    # Number of current YELLOW and RED tokens
    tokens_nb = [y_tokens_set, r_tokens_set]
    # Sort by decreasing order of the number of features and remove columns without any None feature
    sorted_indexes = sorted([key for key, val in none_dict.items() if val[-1]], key=lambda x: none_dict[x][-1], reverse=True)
    # Generate (part of) states
    samples = []
    rec_c4_product([], 0, samples, tokens_nb, sorted_indexes, none_dict, n, max_tokens, env, [])

    v[i] = tmp_value

    #print('v[i] after sampling: {}'.format(v[i]))
    return samples

#  Produce each combination of none features values and store the valid states (recursive function)
#  Input: current list of product of none features values of column(s) (int list list), index of the current column
#  studied (int), list of valid states (int list list list), current number of both players tokens (int list),
#  column indexes (int list), column dictionary (column dict), maximal number of samples (int), maximal number of token
#  of one color (int list), environment (Connect4), indexes of winning column positions (int list list)
#  Output: stop function (bool)
def rec_c4_product(cols_prod, i, result, tokens_nb, col_indexes, col_dict, n, limit, env, col_win_idx):
    # stop case: state found (additional tests)
    if i >= len(col_indexes):
        is_valid_state, p_state = valid_state(cols_prod, col_dict, col_indexes, tokens_nb, env, col_win_idx)
        if is_valid_state: result.append(p_state)
        return True

    else:
        for prod in c4_product(col_dict[col_indexes[i]][0], repeat=col_dict[col_indexes[i]][-1]):
            # condition: valid number of tokens and no 'flying' token
            #print('replacement values: {} for column {}'.format(prod, col_indexes[i]))
            valid, nb_y, nb_r, win_idx = valid_column(limit, prod, tokens_nb, col_dict[col_indexes[i]][2], col_indexes[i])
            #print('i value after valid_column test: {}'.format(i))
            #print('prod: {} - column: {}'.format(prod, col_dict[col_indexes[i]][2]))

            if valid:

                cols_prod.append(prod)

                if win_idx: col_win_idx.append(win_idx)
                #print('col_win_idx before rec call: {}'.format(col_win_idx))
                rec_c4_product(cols_prod, i + 1, result, [tokens_nb[0] + nb_y, tokens_nb[1] + nb_r], col_indexes, col_dict, n, limit, env, col_win_idx)
                if win_idx: del col_win_idx[-1]
                #print('col_win_idx after rec call: {}'.format(col_win_idx))

                # specific case: only n samples are required
                if 0 < n == len(result): return True
                del cols_prod[-1]

        return True

#  Check the column validity (e.g. no 'flying' tokens)
#  Input: maximal number of token of one color (int list), none features values (int list), current number of both players
#  tokens (int list), column under study (int list), column index (int)
#  Output: validity variable (bool), number of added yellow/red tokens (int), indexes of winning rows (int list list)
def valid_column(limit, values, tokens_nb, column, col_idx):

    def update_current_color(color, cpt, new_color, tmp_win_col_idx, li_idx, col_idx):
        #print('color', color, cpt, new_color)
        if color == new_color:
            cpt += 1
            tmp_win_col_idx.append([li_idx, col_idx])
            if cpt > 4:
                #print('Not feasible terminal state! too many tokens in a column')
                return False, color, cpt, tmp_win_col_idx

            return True, color, cpt, tmp_win_col_idx
        else:
            tmp_win_col_idx = [[li_idx, col_idx]]
            color = new_color
            cpt = 1
            return True, color, cpt, tmp_win_col_idx

    nb_y = 0
    nb_r = 0
    i = 1
    empty_cell = False
    current_color = None
    current_color_cpt = 0
    tmp_win_col_idx = []
    win_col_idx = None

    # loop on reverse order (when an empty cell is encountered, the next values must be empty as well)

    # count the new tokens
    for idx in range(len(column)-1, -1,-1):
        #print(idx)
        if column[idx] == EMPTY:
            empty_cell = True

        elif column[idx] in TOKENS:
            # flying token
            if empty_cell:
                #print('flying token: {}'.format(column))
                #print('values: {}'.format(values))
                return False, nb_y, nb_r, []

            # avoid terminal state with more than 4 subsequent tokens in a column
            valid, current_color, current_color_cpt, tmp_win_col_idx = update_current_color(current_color, current_color_cpt, column[idx], tmp_win_col_idx, idx, col_idx)
            #print('color', current_color, current_color_cpt)

            if not valid:
                return valid, nb_y, nb_r, []

            if len(tmp_win_col_idx) == 4:
                #print('Wining column found: column {} {} - idx {} - color {}'.format(col_idx, column, tmp_win_col_idx, current_color))
                tmp_win_col_idx.append(current_color)
                win_col_idx = deepcopy(tmp_win_col_idx)

        elif column[idx] is None:
            if values[-i] == EMPTY:
                empty_cell = True

            else:
                if empty_cell:
                    #print('flying token: {}'.format(column))
                    #print('values: {}'.format(values))
                    return False, nb_y, nb_r, []

                elif values[-i] in TOKENS:

                    # avoid terminal state with more than 4 subsequent tokens in a column
                    valid, current_color, current_color_cpt, tmp_win_col_idx = update_current_color(current_color, current_color_cpt,
                                                                                   values[-i], tmp_win_col_idx, idx, col_idx)
                    if not valid:
                        #if current_color == RED: print('too many {} tokens: col {}  values {}'.format(current_color, column, values))
                        return valid, nb_y, nb_r, []

                    if values[-i] == YELLOW:
                        nb_y += 1

                    else:
                        nb_r += 1

                    # a winning colum is found
                    if len(tmp_win_col_idx) == 4:
                        #print('Wining column found: column {} {} - values {} - idx {} - color {}'.format(col_idx, column, values, tmp_win_col_idx, current_color))
                        tmp_win_col_idx.append(current_color)
                        win_col_idx = deepcopy(tmp_win_col_idx)

            i += 1

    #print('y: {} - r {} - limit {}'.format(tokens_nb[0] + nb_y, tokens_nb[1] + nb_r, limit))
    return tokens_nb[0] + nb_y <= limit and tokens_nb[1] + nb_r <= limit, nb_y, nb_r, win_col_idx

#  Generate the state, check whether the state is terminal/plausible or not
#  Input: none features values (int list list), column dictionary (column dict), indexes of none columns (i.e. column
#  containing at least one none feature) (int list), number of both player tokens (int list), environment (Connect4),
#  indexes of winning columns (int list)
#  Output: validity variable (bool), state (int list list)
def valid_state(values, col_dict, none_col_idx, tokens_nb, env, col_win_idx):
    allow_terminal = True
    invalid = False, [[]]

    #  Different number of tokens
    if tokens_nb[0] != tokens_nb[1]:
        #print('Invalid state: diff number of tokens: y {} - r {} ------------------'.format(tokens_nb[0], tokens_nb[1]))
        return invalid

    # Get column height / Replace None features  ---------------------------------------------------
    column_height = []
    col_dict_copy = deepcopy(col_dict)

    for col in col_dict:
        idx_values = 0
        height = 0

        for idx, feature in enumerate(col_dict_copy[col][2]):
            if feature in TOKENS:
                height += 1

            elif feature is None:
                new_val = values[none_col_idx.index(col)][idx_values]
                #print('replace None feature: col {} - val {}'.format(col_dict_copy[col][2], new_val))
                col_dict_copy[col][2][idx] = new_val
                idx_values += 1
                if new_val in TOKENS:
                    height += 1

        #print('column {}: {}'.format(col, col_dict[col][2]))

        column_height.append(height)

    #print('column_height: {}'.format(column_height))

    # Build the state / Check whether it's a plausible board or not -------------------------------------------
    p_state = [[0 for _ in range(env.cols)] for _ in range(env.rows)]
    cur_token = YELLOW
    idx_pos = [env.rows - 1 for _ in range(env.cols)]

    # Simulate a game
    while any(column_height) != 0:
        # sort column by number of opp token in column
        valid_cols = sorted([col for col in col_dict_copy.keys() if col_dict_copy[col][2][idx_pos[col]] == cur_token and column_height[col]], key= lambda x: col_dict_copy[x][2][:idx_pos[x] + 1].count(cur_token * (-1)), reverse=True)
        #print('valid_cols: {}'.format(valid_cols))

        # stop case: Token can't be in any column
        if not valid_cols:
            #print('Invalid state: not a real game ---------------------')
            return invalid

        # choose column
        found = False
        for col in valid_cols:
            #print('col: {}'.format(col))
            #print('idx_pos: {}'.format(idx_pos[col]))
            #print('subsequent token: {}'.format(col_dict_copy[col][2][idx_pos[col] - 1]))
            if idx_pos[col] and col_dict_copy[col][2][idx_pos[col] - 1] == cur_token * (-1):
                chosen_col = col
                found = True
                #print('valid_cols: {}'.format(valid_cols))
                #print('column height: {}'.format(column_height))
                #print('chosen col: {}'.format(chosen_col))
                break

        # default choice
        if not found:
            chosen_col = valid_cols[0]
        # place token
        p_state[idx_pos[chosen_col]][chosen_col] = cur_token
        # updates
        column_height[chosen_col] -= 1
        idx_pos[chosen_col] -= 1
        cur_token = cur_token * (-1)

    #print('It s a real game!')

    # Remove infeasible terminal state ---------------------------------------------------------------

    # Get the number of wining lines, column, diagonal of both players
    y_idx, r_idx = countrows(p_state, col_win_idx)

    # The state is valid and is not terminal
    if not y_idx and not r_idx:
        return True, p_state

    # One of the player doesn't have any 4 in a row
    elif not y_idx or not r_idx:
        # Get the winner
        win_idx, color = (y_idx, YELLOW) if y_idx else (r_idx, RED)
        #print('Only one player has one/several rows: idx {} - color {}'.format(win_idx, color))
        return is_valid_rows(p_state, win_idx, color, invalid)[:2]

    # Both players have at least one 4 in a row
    else:
        #print('Both player has one/several rows: y_idx {} - r_idx {}'.format(y_idx, r_idx))
        valid, p_state, red_coord = is_valid_rows(p_state, r_idx, RED, invalid)
        if not valid:
            #print('RED NOT VALID')
            return invalid

        else:
            valid, p_state, _ = is_valid_rows(p_state, y_idx, YELLOW, invalid, red_coord=red_coord)
            if not valid:
                #print('YELLOW NOT VALID')
                return invalid

    # Terminal state
    if not allow_terminal and terminal(p_state, env, None):
        #print('It s a terminal state ------------------------')
        #print(p_state)
        return invalid

    #print('VALID ! ------------------------------')
    return True, p_state

#  Check whether the winning rows of a player are valid ones
#  Input: state (int list list list), indexes of winning rows (int list list), token color (int), pre-defined return
#  (bool-list list), position of the last token played by red (player 2) (int list)
#  Output: validity variable (bool) or state (int list list), position of the last token played (int list)
def is_valid_rows(state, idx, color, invalid, red_coord=None):
    # Only one row
    if len(idx) == 1:
        valid, coord = is_last_token(state, idx[0], color, red_coord)
        #if not valid: print('INVALID for color {}'.format(color))
        return valid, invalid[1] if not valid else state, coord

    # Several at the same time
    else:
        intersection = intersect(idx)
        #print('intersection: {}'.format(intersection))

        if len(intersection) == 1:
            valid, coord = is_last_token(state, intersection, color, red_coord)
            return valid, invalid[1] if not valid else state, coord

        else:  # none or several intersection(s) translate an infeasible terminal state
            return invalid[0], invalid[1], None

#  Compute the intersection of several winning rows
#  Input: winning rows (int list list list)
#  Output: list of intersect positions (int list list)
def intersect(coord_lists):
    tmp_coord_lists = []
    for coords in coord_lists:
        tuple_coords = [tuple(elm) for elm in coords]
        tmp_coord_lists.append(tuple_coords)
    #print('tmp_coord_lists: {}'.format(tmp_coord_lists))
    return list(set(tmp_coord_lists[0]).intersection(*tmp_coord_lists))

#  Check whether the token at idx is the last token played in the board or not
#  Input: state (int list list), indexes of the intersection between several winning rows (int list list), token color
#  of winning rows (int), position of the last token played by red (player 2) (int list)
#  Output: valid last token move (bool), position of the last played token (int list)
def is_last_token(p_state, idx, token_color, red_coord=None):
    # case: intersection of length 1 --> only one coord to check -------------------------------
    if len(idx) == 1:
        coord = idx[0]
        top1_line_idx = max(-1, coord[0] - 1)
        #print('top1_line_idx: {}'.format(top1_line_idx))

        # coord is not located at the top line
        if top1_line_idx >= 0:
            # case: yellow token
            if token_color == YELLOW:
                if p_state[top1_line_idx][coord[1]] == EMPTY:
                    return True, None
                # a yellow token above the intersection
                elif p_state[top1_line_idx][coord[1]] == YELLOW:
                    return False, None
                elif p_state[top1_line_idx][coord[1]] == RED:
                    # last red token played is at red_coord
                    if red_coord is not None:
                        #print('compare with: {}'.format([top1_line_idx, coord[1]]))
                        return [top1_line_idx, coord[1]] in red_coord, None

                    else:
                        top2_line_idx = max(-1, coord[0] - 2)
                        #print('top2_line_idx: {}'.format(top2_line_idx))
                        if top2_line_idx >= 0:
                            return p_state[top2_line_idx][coord[1]] == EMPTY, None

                        else:
                            return True, None

            # case: red token
            else:
                return (True, coord) if p_state[top1_line_idx][coord[1]] == EMPTY else (False, None)

        # coord is already at the top of the board, it's a valid move
        else:
            return True, coord if token_color == RED else None

    # case: one row to check --------------------------------------------------------
    else:
        is_column = len(idx) == [i[1] for i in idx].count(idx[0][1])
        # column, only the token at the top must be checked
        if is_column:
            #print('row is column: {} focus only on position {}'.format(idx, idx[-1]))
            coord = idx[-1]
            return is_last_token(p_state, [coord], token_color, red_coord)

        # line/diagonal, several tokens must be checked
        else:

            # get the indexes to check
            remove_cpt = len(idx) - 4
            check_indexes = idx if not remove_cpt else idx[remove_cpt:-remove_cpt]
            #print('check_indexes: {}'.format(check_indexes))
            valid_list = []

            for i in check_indexes:
                valid, coord = is_last_token(p_state, [i], token_color, red_coord)
                valid_list.append(valid)
                if coord:
                    if red_coord is None:
                        red_coord = [coord]
                    else:
                        red_coord.append(coord)

            if valid_list.count(True):
                return True, red_coord if token_color == RED else None

            return False, None

#  Count winning lines / columns / diagonals of 4 tokens at least (for each player)
#  Input: state (int list list), indexes of winning column (int list)
#  Output: indexes of winning lines / columns / diagonals for both players (int list list)
def countrows(state, col_win_idx):
    y_win_idx = [] # indexes of row of 4 YELLOW tokens
    r_win_idx = [] # indexes of row of 4 RED tokens
    cols = len(state[0])
    lines = len(state)

    # Get the already computed winning columns ----------------------
    for row in col_win_idx:
        if row[-1] == YELLOW:
            y_win_idx.append(row[:-1])
        else:
            r_win_idx.append(row[:-1])

    # Search for the wining lines --------------------------------------
    for l in range(lines):
        idx = []
        color = None

        for c in range(cols):
            color, idx, y_win_idx, r_win_idx = update_winning_rows(state, l, c, color, y_win_idx, r_win_idx, idx)

        # last possible row of the line is found
        color, idx, y_win_idx, r_win_idx = update_winning_rows([], -1, -1, color, y_win_idx, r_win_idx, idx, 'last')


    # Search for the wining diagonals --------------------------------------
    for l in range(lines):
        if not l or l == lines - 1: # extreme lines: several positions to search for diagonals
            for c in range(cols//2 + 1):
                # get range and slope
                diag_range, slope_orientation = get_diagonal_range(l, c, lines, cols)
                idx = []
                color = None

                #print('line: ', l, 'column: ', c,  'diag_range: ', diag_range, 'slope_orientation: ', slope_orientation)
                for i in range(diag_range):
                    tmp_col = c + i
                    tmp_li = l + i if slope_orientation == 1 else l - i
                    color, idx, y_win_idx, r_win_idx = update_winning_rows(state, tmp_li, tmp_col, color, y_win_idx, r_win_idx,
                                                                           idx)

                # last possible row of the diagonal is found
                color, idx, y_win_idx, r_win_idx = update_winning_rows([], -1, -1, color, y_win_idx, r_win_idx, idx,
                                                                       'last')

        else: # only one position

            # get range and slope
            diag_range, slope_orientation = get_diagonal_range(l, 0, lines, cols)
            idx = []
            color = None

            #print('line: ', l, 'column: ', 0, 'diag_range: ', diag_range, 'slope_orientation: ', slope_orientation)
            for i in range(diag_range):
                tmp_li = l + i if slope_orientation == 1 else l - i
                color, idx, y_win_idx, r_win_idx = update_winning_rows(state, tmp_li, i, color, y_win_idx, r_win_idx, idx)

            # last possible row of the diagonal is found
            color, idx, y_win_idx, r_win_idx = update_winning_rows([], -1, -1, color, y_win_idx, r_win_idx, idx, 'last')

    return y_win_idx, r_win_idx

#  Get the diagonal range according to x,y board position
#  Input: line/column index, total number of lines/columns (int)
#  Output: diagonal range (int list), slope type (positive/negative) (int)
def get_diagonal_range(line, column, lines, cols):
    # l: 0, 1, 2 --> positively sloped diagonal
    # l: 3, 4, 5 --> negatively sloped diagonal
    slope = 1 if line < lines // 2 else -1
    if not column:
        if slope == 1:
            diagonal_range = cols - (line + 1)
        else:
            diagonal_range = line + 1
    else:
        diagonal_range = cols - column

    return diagonal_range, slope

#  Update possible winning rows from a specific board position
#  Input: state (int list list), x,y position (int), current color (int), list of winning rows of both players
#  (int list list), indexes of current row (int list list),  last possible row of a line/diagonal (str)
#  Output: current color (int), indexes of current row (int list list), list of winning rows of both players
#  (int list list)
def update_winning_rows(state, l, c, color, y_win_idx, r_win_idx, idx, last=None):
    if last is None:
        if color is not None:
            if state[l][c] == color:
                # print('add element')
                idx.append([l, c])
            else:
                # print('change of color')

                # a row is found
                if len(idx) >= 4 and color != EMPTY:
                    #print('Row of {} {} tokens found: {}'.format(len(idx), color, idx))
                    y_win_idx.append(idx) if color == YELLOW else r_win_idx.append(idx)

                # reset coords / color
                idx = [[l, c]]
                color = RED if state[l][c] == RED else YELLOW if state[l][c] == YELLOW else EMPTY
                # print('new color: {}'.format(color))

        # color was none due to init or previous EMPTY
        else:
            color = state[l][c]
            idx = [[l, c]]
    else:
        if len(idx) >= 4 and color != EMPTY:
            #print('Last Row of {} {} tokens found: {}'.format(len(idx), color, idx))
            y_win_idx.append(idx) if color == YELLOW else r_win_idx.append(idx)

    return color, idx, y_win_idx, r_win_idx

#  From flattened state to state
#  Input: flattened state (int list), environment (Connect4)
#  Output: state (int list list)
def to_state(flat_state, env):
    state = []
    line = []

    for feature in flat_state:
        line.append(feature)
        if len(line) == env.cols:
            state.append(line)
            line = []

    #print('to_state function test')
    #print('not flat state: {}'.format(state))
    return state

#  From state to flattened state
#  Input: state (int list list)
#  Output: flattened state (int list)
def to_flattened_state(state):
    flat_state = []

    for elm in state:
        flat_state.extend(elm)
    #print('test to flattened state: {}'.format(flat_state))
    return flat_state

#  Specific product which avoids combinations with 'flying' tokens
#  Input: list of possible feature values for a board column (int list), number of 'none' features
#  of a board column (int)
#  Output: iterator of combinations of feature values (int list iterator)
def c4_product(*args, repeat=1):
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]

    for pool in pools:
        result = [x + [y] for x in result for y in pool if not (len(x) and y == 0 and x[-1] != 0)]
        #print('result: {}'.format(result))

    #print('combinations: {}'.format(result))
    for prod in result:
        yield tuple(prod)

#  Get available actions
#  Input: state (int list list), environment (Connect4)
#  Output: actions list (int list)
def get_actions(s, env):
    if type(s) != list: print(s)
    return [i for i in range(env.cols) if s[0][i] == 0]

### Render HXP ###

#  Display the (B-)HXP to the user
#  Input: important state-action list (state-action list), environment (Connect4), agent, importance type (str),
#  runtime (float), additional information (dict)
#  Output: None
def render(hxp, env, agent, imp_type, runtime, add_info):
    # Render
    env_copy = deepcopy(env)
    for s_a_list, i in hxp:
        print("Timestep {}".format(i))
        env_copy.board = s_a_list[0]
        env_copy.render()
        print('Action {}'.format(s_a_list[1]))
        if imp_type == 'transition':
            env_copy.board = s_a_list[2]
            env_copy.render()

    # Runtime
    print("-------------------------------------------")
    print("Explanation achieved in: {} second(s)".format(runtime))
    print("-------------------------------------------")
    return

### Predicates ###

#  Check whether player 1 wins
#  Input: state (int list list), additional information (dictionary)
#  Output: (bool)
def win(s, info):
    env, agent = info['env'], info['agent']
    env.board = s
    return env.win(agent.token)

#  Check whether player 1 loses
#  Input: state (int list list), additional information (dictionary)
#  Output: (bool)
def lose(s, info):
    env, agent = info['env'], info['agent']
    env.board = s
    # specific case for sampled states (paxp computation) ith the same amount of tokens of each player
    if info.get('lose_weak_paxp') is not None:
        return not env.win(agent.token) and env.win(agent.token * (-1))

    else:
        return env.win(agent.token * (-1))

#  Check whether player 1 obtained more tokens in the mid-column or not in comparison with the initial history's state
#  Input: state (int list list), additional information (dictionary)
#  Output: (bool)
def control_midcolumn(s, info):
    agent, initial_s = info['agent'], info['initial_s']
    #print(initial_s)

    advantage_initial_s = [row[len(s) // 2] for row in initial_s].count(agent.token) - [row[len(s) // 2] for row in initial_s].count(agent.token * (-1))
    advantage_s = [row[len(s) // 2] for row in s].count(agent.token) - [row[len(s) // 2] for row in s].count(agent.token * (-1))
    return advantage_s > advantage_initial_s

#  Check whether player 1 obtained more '3 tokens in a row' or not in comparison with the initial history's state
#  Input: state (int list list), additional information (dictionary)
#  Output: (bool)
def align3(s, info):
    agent, initial_s = info['agent'], info['initial_s']
    return count3inarow(s, agent.token) > count3inarow(initial_s, agent.token)

#  Check whether player 1 succeeds in avoiding player 2 to get more '3 tokens in a row' or not
#  Input: state (int list list), additional information (dictionary)
#  Output: (bool)
def counteralign3(s, info):
    agent, initial_s = info['agent'], info['initial_s']
    opponent_token = agent.token * (-1)
    return count3inarow(s, opponent_token) == count3inarow(initial_s, opponent_token)

#  Count number of 3 in a row from a board for 1 player
#  Input: board (int list list) and token of the agent (int)
#  Reward: number of 3 in a row (int)
def count3inarow(s, token):
    cols = len(s[0])
    rows = len(s)
    cpt = 0
    # Check horizontal locations for win
    for c in range(cols - 2):
        for r in range(rows):
            if s[r][c] == token and s[r][c + 1] == token and s[r][c + 2] == token:
                cpt += 1

    # Check vertical locations for win
    for c in range(cols):
        for r in range(rows - 2):
            if s[r][c] == token and s[r + 1][c] == token and s[r + 2][c] == token:
                cpt += 1

    # Check positively sloped diagonals
    for c in range(cols - 2):
        for r in range(rows - 2):
            if s[r][c] == token and s[r + 1][c + 1] == token and s[r + 2][c + 2] == token:
                cpt += 1

    # Check negatively sloped diagonals
    for c in range(cols - 2):
        for r in range(3, rows):
            if s[r][c] == token and s[r - 1][c + 1] == token and s[r - 2][c + 2] == token:
                cpt += 1

    return cpt

#  Check whether a predicate defined via PAXp holds or not
#  Input: state (int list list), additional information (dictionary)
#  Output: (bool)
def redefined_predicate(s, info):
    predicate = info['redefined_predicate']
    #print('predicate : {}'.format(predicate))
    #print('state: {}'.format(s))

    for i in range(len(predicate)):
        for j in range(len(predicate[0])):
            feature = predicate[i][j]
            if feature is not None and feature != s[i][j]:
                #print('difference pred {} -- state {}'.format(feature, s[i][j]))
                return False

    return True

### Find histories for a specific predicate ###

#  Verify if the last state from a proposed history respects a predicate
#  Input: state (int list list), predicate (str), additional information (dict)
#  Output: (bool)
def valid_history(s, predicate, info):
    if predicate == 'win':
        return win(s, info)
    elif predicate == 'lose':
        return lose(s, info)
    elif predicate == 'control_midcolumn':
        return control_midcolumn(s, info)
    elif predicate == 'align3':
        return align3(s, info)
    else:
        return counteralign3(s, info)
