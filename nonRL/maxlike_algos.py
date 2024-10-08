from typing import Literal, Union
from buckshot_roulette.ai import AbstractEngine
from buckshot_roulette.game import BuckshotRoulette, Items
from dataclasses import astuple
import numpy as np
from copy import copy
import random
from functools import lru_cache
import hashlib
import json
from tqdm import tqdm
from typing import List, Tuple

expectimax_cache = {}

def make_hashable(*args, **kwargs):
    """Convert input arguments to a hashable format."""
    args_str = json.dumps(args, sort_keys=True)
    kwargs_str = json.dumps(kwargs, sort_keys=True)
    return hashlib.sha256((args_str + kwargs_str).encode()).hexdigest()
#progress = tqdm()
def global_cache(maxsize=128):
    def decorator(func):
        cache = lru_cache(maxsize=maxsize)

        def wrapped(*args, **kwargs):
            hashable_args = make_hashable(args, kwargs)
            #return cache(lambda _: func(*args, **kwargs))(hashable_args)
            
            # if not hashable_args in expectimax_cache:
            #     expectimax_cache[hashable_args] = func(*args, **kwargs)

            # return expectimax_cache[hashable_args]
            #progress.update()
            return func(*args, **kwargs)

        return wrapped
    return decorator

class Expectimax(AbstractEngine):
    def __init__(self, playing_as: Literal[0, 1]):
        self.known_shells: list[bool | None] = []
        self.planned_moves = []
        self.inverter_uncertainty = False # True if algo has used inverter and hasn't shot yet
        super().__init__(playing_as)
    
    def choice(self, board: BuckshotRoulette):
        diff = len(self.known_shells) - len(board._shotgun)
        if diff > 0:
            # Less bullets in the shotgun than expected; likely shot
            self.known_shells = self.known_shells[diff:]
        elif diff < 0:
            # More bullets in the shotgun than expected; likely due to a reset
            self.known_shells = [None] * len(board._shotgun)
        
        moves = board.moves()
        own_moves = moves.copy()
        if 'adrenaline' in moves:
            moves.remove('adrenaline') # Adrenaline used automatically
            for item in board.items[1 - self.me]:
                if item not in moves:
                    moves.append(item)
        
        if len(self.planned_moves) > 0:
            return self.planned_moves.pop(0)
        
        out_move = None
        
        # Hard Rules
        if 'cigarettes' in moves:
            moves.remove('cigarettes')
            if board.charges[self.me] < board.max_charges:
                out_move = 'cigarettes'
        if 'burner_phone' in moves:
            moves.remove('burner_phone')
            if len(board._shotgun) > 1 and None in self.known_shells:
                out_move = 'burner_phone'         
        if 'meds' in moves:
            moves.remove('meds')
            if 'cigarettes' in moves and board.max_charges - board.charges[self.me] > 1:
                # Cancel out risk, take the 50/50 for 2 lives
                out_move = 'cigarettes'
                self.planned_moves.append('meds')
        
        if out_move == None:
            # Generate repeating set of moves            
            em_moves = ['op', 'self'] if 'op' in moves else []
            
            for item in board.items[self.me]:
                if item in own_moves and item not in ['adrenaline', 'cigarettes', 'burner_phone', 'meds']:
                        em_moves.extend([item] * board.items[self.me][item])
            em_moves = sorted(em_moves)
            em_ad_moves = []
            if 'adrenaline' in own_moves:
                for item in board.items[1-self.me]:
                    if item in own_moves and item not in ['adrenaline', 'cigarettes', 'burner_phone', 'meds']:
                        em_ad_moves.extend([item] * min(board.items[1-self.me][item], board.items[self.me].adrenaline))
            em_ad_moves = sorted(em_ad_moves)
            # Generate the entire next move sequence up until the next shot
            # (For handcuffs, we will run this again after the first shot)
            self.planned_moves = self.expectimax(
                own_moves=em_moves,
                op_moves=em_ad_moves,
                adrenaline_count=board.items[self.me].adrenaline,
                info=board.shotgun_info()
            )
            out_move = self.planned_moves[0]
            self.planned_moves = self.planned_moves[1:]
        
        
        if out_move in own_moves:            
            return out_move
        else:            
            if 'adrenaline' in own_moves:
                self.planned_moves.append(out_move)
                return 'adrenaline'
            else:
                # Invalid state
                # print('Invalid!')
                return random.choice(board.moves())  
    
    
    def expectimax(self, own_moves: list[str], op_moves: list[str], adrenaline_count: int, info: tuple[int, int]):
        #@global_cache(maxsize=1024)
        def search(
            own_moves: List[str], 
            op_moves: List[str], 
            adrenaline_count: int, 
            prob_map: List[float],
            pos: int = 0,
            is_hc: bool = False,
            depth=8
        ) -> Tuple[float, List[str]]:
            nonlocal self
            if depth == 0 or len(prob_map) == 0 or pos >= len(self.known_shells) or (len(own_moves) + len(op_moves)) == 0:
                return 0, []
            
            best_val = float('-inf')
            best_moves = []
            
            def update(val, moves):
                nonlocal best_val, best_moves
                if val > best_val or (val == best_val and len(moves) < len(best_moves)):
                    best_val = val
                    best_moves = moves

            for move in own_moves:
                next_moves = own_moves.copy()
                if move not in ['op', 'self']:
                    next_moves.remove(move)
                
                if move == 'op':
                    if is_hc:
                        next_map = prob_map[1:]  # Correct slicing
                        val, moves = search(next_moves, op_moves, adrenaline_count, next_map, pos + 1, False, depth-1)
                        update(prob_map[0] + val, ['op'] + moves)
                    else:
                        update(prob_map[0], ['op'])
                elif move == 'self':
                    if is_hc:
                        # next_map = prob_map[1:]  # Correct slicing
                        
                        # live_val, live_moves = search(next_moves, op_moves, adrenaline_count, next_map, pos + 1, False, depth-1) if prob_map[0] != 0.0 else (0, None)
                        # blank_val, blank_moves = search(next_moves, op_moves, adrenaline_count, next_map, pos + 1, True, depth-1) if prob_map[0] != 1.0 else (0, None)
                        # live_val -= 1
                        # combined_val = (prob_map[0] * live_val) + ((1 - prob_map[0]) * blank_val)
                        # update(combined_val, ['self'])
                        update(1 - prob_map[0], ['self'])
                    else:
                        update(1 - prob_map[0], ['self'])
                elif move == 'handcuffs':
                    val, moves = search(next_moves, op_moves, adrenaline_count, prob_map, pos, True, depth-1)
                    update(val, ['handcuffs'] + moves)
                elif move == 'magnifying_glass':
                    live_map = prob_map.copy()
                    live_map[0] = 1.0
                    
                    blank_map = prob_map.copy()
                    blank_map[0] = 0.0
                    
                    val_live, moves_live = search(next_moves, op_moves, adrenaline_count, live_map, pos, is_hc, depth-1)
                    val_blank, moves_blank = search(next_moves, op_moves, adrenaline_count, blank_map, pos, is_hc, depth-1)
                    
                    combined_val = (prob_map[0] * val_live) + ((1 - prob_map[0]) * val_blank)
                    update(combined_val, ['magnifying_glass'])
                elif move == 'beer':
                    next_map = prob_map[1:]  # Correct slicing
                    val, moves = search(next_moves, op_moves, adrenaline_count, next_map, pos + 1, is_hc, depth-1)
                    update(val, ['beer'] + moves)
                elif move == 'saw':
                    val, moves = search(next_moves, op_moves, adrenaline_count, prob_map, pos, is_hc, depth-1)
                    update(prob_map[0] + val, ['saw'] + moves)
                elif move == 'inverter':
                    next_map = prob_map.copy()
                    next_map[0] = 1 - next_map[0]
                    
                    val, moves = search(next_moves, op_moves, adrenaline_count, next_map, pos, is_hc, depth-1)
                    update(val, ['inverter'] + moves)
            
            if adrenaline_count > 0:
                for move in op_moves:
                    next_moves = op_moves.copy()
                    if move not in ['op', 'self']:
                        next_moves.remove(move)
                    
                    if move == 'handcuffs':
                        val, moves = search(own_moves, next_moves, adrenaline_count - 1, prob_map, pos, True, depth-1)
                        update(val, ['adrenaline', 'handcuffs'] + moves)
                    elif move == 'magnifying_glass':
                        live_map = prob_map.copy()
                        live_map[0] = 1.0
                        
                        blank_map = prob_map.copy()
                        blank_map[0] = 0.0
                        
                        val_live, moves_live = search(own_moves, next_moves, adrenaline_count - 1, live_map, pos, is_hc, depth-1)
                        val_blank, moves_blank = search(own_moves, next_moves, adrenaline_count - 1, blank_map, pos, is_hc, depth-1)
                        
                        combined_val = (prob_map[0] * val_live) + ((1 - prob_map[0]) * val_blank)
                        update(combined_val, ['adrenaline', 'magnifying_glass'])
                    elif move == 'beer':
                        next_map = prob_map[1:]  # Correct slicing
                        val, moves = search(own_moves, next_moves, adrenaline_count - 1, next_map, pos + 1, is_hc, depth-1)
                        update(val, ['adrenaline', 'beer'] + moves)
                    elif move == 'saw':
                        val, moves = search(own_moves, next_moves, adrenaline_count - 1, prob_map, pos, is_hc, depth-1)
                        update(prob_map[0] + val, ['adrenaline', 'saw'] + moves)
                    elif move == 'inverter':
                        next_map = prob_map.copy()
                        next_map[0] = 1 - next_map[0]
                        
                        val, moves = search(own_moves, next_moves, adrenaline_count - 1, next_map, pos, is_hc, depth-1)
                        update(val, ['adrenaline', 'inverter'] + moves)
            
            return best_val, best_moves
        
        X, N = info
        X -= self.known_shells.count(True)
        N -= self.known_shells.count(True) + self.known_shells.count(False)
        prob_map = [X/N if shell == None else float(shell) for shell in self.known_shells]
        
        val, moves = search(own_moves, op_moves, adrenaline_count, prob_map, 0, False, depth=len(own_moves) + len(op_moves))
        return moves
    
    def post(self, last_move, result):
        if last_move in ['op', 'self', 'beer']:
            self.known_shells = self.known_shells[1:]
            self.inverter_uncertainty = False
        match last_move:
            case 'inverter':
                if self.known_shells[0] != None:
                    self.known_shells[0] = not self.known_shells[0]
                else:
                    self.inverter_uncertainty = True
            case 'magnifying_glass':
                self.known_shells[0] = result
                self.inverter_uncertainty = False
            case 'burner_phone':
                if result != None:                    
                    self.known_shells[result[0]] = result[1]
    def on_reload(self):
        self.known_shells = []

class OptimalAlgo(AbstractEngine):
    def __init__(self, playing_as: Literal[0, 1]):
        self.known_shells: list[bool | None] = []
        self.planned_move = None
        self.inverter_uncertainty = False # True if algo has used inverter and hasn't shot yet
        super().__init__(playing_as)   
        
    def minimize_criticality(self, board: BuckshotRoulette):
        def criticality(op_items: Items):
            X, N = board.shotgun_info()
            max_hits = 1
            if X == 0:
                if op_items.inverter > 0:
                    max_hits = 1 if op_items.handcuffs == 0 else min(2, min(op_items.inverter, N))
                else:
                    return 0
            else:
                max_hits = 1 if op_items.handcuffs == 0 else min(2, X + min(op_items.inverter, N))
                
            max_dmg = max_hits
            max_dmg += max(op_items.saw, max_hits)        
            return min(max_dmg, board.charges[self.me])
        
        op_items = board.items[1 - self.me]
        baseline = criticality(op_items)
        impacts = {}
        for item in ['saw', 'handcuffs', 'inverter']:
            oi = copy(op_items)
            oi[item] -= 1
            impacts[item] = baseline - criticality(oi)
        
        best = sorted(impacts, key=impacts.get)
        return best, impacts, baseline
    def choice(self, board: BuckshotRoulette):
        diff = len(self.known_shells) - len(board._shotgun)
        if diff > 0:
            # Less bullets in the shotgun than expected; likely shot
            self.known_shells = self.known_shells[diff:]
        elif diff < 0:
            # More bullets in the shotgun than expected; likely due to a reset
            self.known_shells = [None] * len(board._shotgun)
        
        moves = board.moves()
        own_moves = moves.copy()
        if 'adrenaline' in moves:
            moves.remove('adrenaline') # Adrenaline used automatically
            for item in board.items[1 - self.me]:
                if item not in moves:
                    moves.append(item)
        if board._active_items.saw > 0 and 'self' in moves:
            moves.remove('self')
        
        if self.planned_move != None:
            move = self.planned_move
            self.planned_move = None
            if move in moves:                
                # Should always be true, but if not we should just move on
                return move
        
        # Hard Rules
        out_move = None      
        
        X, N = board.shotgun_info()
        unknown_round_chance: float | None = X / N
        if not self.inverter_uncertainty:
            # Resolve shells
            live, blank = X, N-X            
            for shell in self.known_shells:
                if shell != None:
                    if shell:
                        live -= 1
                    else:
                        blank -= 1
            
            if live == 0:
                # No unknown live rounds left
                # All unknown rounds are blank
                unknown_round_chance = None
                self.known_shells = [False if shell == None else shell for shell in self.known_shells]
                
            elif blank == 0:
                # No unknown blank rounds left
                # All unknown rounds are live
                unknown_round_chance = None
                self.known_shells = [True if shell == None else shell for shell in self.known_shells]
            else:
                unknown_round_chance = live / (blank+live)
        else:
            # Escape inverter uncertainty as quickly as possible
            # out_move = 'op'
            pass
        
        # Criticality Eval
        crit_moves, crit_diffs, crit_baseline = self.minimize_criticality(board)
        if 'adrenaline' in moves and crit_baseline == board.charges[self.me]:
            for cm in crit_moves:
                if crit_diffs[cm] > 0 and cm in moves:
                    out_move = cm        
        
        items = board.items[self.me]
        # Always use 
        if out_move == None:
            if 'meds' in moves:
                moves.remove('meds')
                if 'cigarettes' in moves and board.max_charges - board.charges[self.me] > 1:
                    # Cancel out risk, take the 50/50 for 2 lives
                    out_move = 'cigarettes'
                    self.planned_move = 'meds'
            if 'beer' in moves:
                # If we can reach a guaranteed live shell, go for that
                if True in self.known_shells:
                    idx = self.known_shells.index(True)
                    if items.beer >= idx and idx != 0:
                        out_move = 'beer'
                if 'inverter' in moves and False in self.known_shells:
                    idx = self.known_shells.index(False)
                    if items.beer >= idx != 0:
                        out_move = 'beer'
            if 'burner_phone' in moves:
                moves.remove('burner_phone')
                if len(board._shotgun) > 1 and None in self.known_shells:
                    out_move = 'burner_phone'            
            if 'handcuffs' in moves:        
                moves.remove('handcuffs')    
                if len(board._shotgun) > 1:
                    out_move = 'handcuffs'
            if 'inverter' in moves:
                if self.known_shells[0] == False:
                    out_move = 'inverter'
            if 'saw' in moves and self.known_shells[0] == True:
                out_move = 'saw'
            if 'magnifying_glass' in moves:
                # moves.remove('magnifying_glass')
                # if self.known_shells[0] == None:
                #     out_move = 'magnifying_glass'
                pass  
            if 'cigarettes' in moves:
                moves.remove('cigarettes')
                if board.charges[self.me] < board.max_charges:
                    out_move = 'cigarettes'
        
        if out_move == None and not self.inverter_uncertainty:
            X, N = board.shotgun_info()
            out_move, _ = self.move_tree(moves, X, N, board.items[self.me])
        
        if out_move in own_moves:            
            return out_move
        else:            
            if 'adrenaline' in own_moves:
                self.planned_move = out_move
                return 'adrenaline'
            else:
                # Invalid state
                # print('Invalid!')
                return random.choice(board.moves())
    
    def move_tree(self, moves: list[str], X: int, N: int, items: Items, depth: int = 0, known: bool | None = None):
        # Never cigarettes or handcuffs
        # Never inverter uncertainty
        
        unk_X = X - self.known_shells[depth:].count(True)
        unk_N = N - self.known_shells[depth:].count(True) - self.known_shells[depth:].count(False)
        unknown_bullets = (unk_X, unk_N)        
        if depth < len(self.known_shells) and self.known_shells[depth] != None:
            hit_prob = 1.0 if self.known_shells[depth] else 0.0
            known = self.known_shells[depth]
        elif known != None:
            hit_prob = 1.0 if known else 0.0
        else:
            hit_prob = unknown_bullets[0] / unknown_bullets[1]    
        
        
        if hit_prob == 1:
            if 'saw' in moves:
                return 'saw', 1
            elif 'op' in moves:
                return 'op', 1
        if hit_prob == 0 and 'self' in moves:
            return 'self', 1
        if (unk_N == 1 or hit_prob == 1) and 'beer' in moves:
            moves.remove('beer')            
        
        if depth > 8:
            return 'op', hit_prob
        
        move_vals = {}
        
        for move in moves:
            next_moves: list[str] = moves.copy()
            next_items: Items = copy(items)
            if not move in ['op', 'self'] and next_items[move] <= 0:
                next_moves.remove(move)
                continue
            match move:
                case 'op':
                    move_vals[move] = hit_prob
                case 'self':
                    move_vals[move] = 1 - hit_prob
                case 'saw':
                    #next_moves.remove('saw')
                    next_items.saw -= 1
                    next_moves.remove('self') if "self" in next_moves else 0
                    move_vals[move] = 2 * self.move_tree(next_moves, X, N, next_items, depth, known)[1]
                case 'beer':                    
                    next_items.beer -= 1                    
                    
                    live_case = self.move_tree(next_moves, X-1, N-1, next_items, depth+1, known)[1]
                    blank_case = self.move_tree(next_moves, X, N-1, next_items, depth+1, known)[1]
                    
                    move_vals[move] = (hit_prob * live_case) + ((1 - hit_prob) * blank_case)
                case 'inverter':
                    #next_moves.remove('inverter')
                    next_items.inverter -= 1
                                        
                    live_case = self.move_tree(next_moves, X-1, N, next_items, depth, known)[1]
                    blank_case = self.move_tree(next_moves, X+1, N, next_items, depth, known)[1]
                    
                    move_vals[move] = (hit_prob * live_case) + ((1 - hit_prob) * blank_case)
                case 'magnifying_glass':
                    #next_moves.remove('magnifying_glass')
                    next_items.magnifying_glass -= 1
                                        
                    live_case = self.move_tree(next_moves, X, N, next_items, depth, known=True)[1]
                    blank_case = self.move_tree(next_moves, X, N, next_items, depth, known=False)[1]
                    
                    move_vals[move] = (hit_prob * live_case) + ((1 - hit_prob) * blank_case)
        if len(move_vals) == 0:
            # Indeterminate- never go here
            return None, 0
        max_move = max(move_vals, key=move_vals.get)
        return max_move, move_vals[max_move]        
        
    def post(self, last_move, result):            
        if last_move in ['op', 'self', 'beer']:
            self.known_shells = self.known_shells[1:]
            self.inverter_uncertainty = False
        match last_move:
            case 'inverter':
                if self.known_shells[0] != None:
                    self.known_shells[0] = not self.known_shells[0]
                else:
                    self.inverter_uncertainty = True
            case 'magnifying_glass':
                self.known_shells[0] = result
                self.inverter_uncertainty = False
            case 'burner_phone':
                if result != None:                    
                    self.known_shells[result[0]] = result[1]
    
    def on_reload(self):
        self.known_shells = []
    
    def find_best_move(self, moves: list[str], board: BuckshotRoulette) -> Literal['op', 'self', 'handcuffs', 'magnifying_glass', 'beer', 'cigarettes', 'saw', 'inverter', 'burner_phone', 'meds', 'adrenaline']:
        # Never cigarettes or handcuffs or magnifying_glass
        # Never inverter uncertainty
        X, N = board.shotgun_info()
        
        if X == N or self.known_shells[0] == True:
            if 'saw' in moves:
                return 'saw'
            elif 'op' in moves:
                return 'op'
        if X == 0 or self.known_shells[0] == False and 'self' in moves:
            return 'self'
        if len(board._shotgun) == 1 and 'beer' in moves:
            moves.remove('beer')
        
        move_vals = {}
        for move in moves:
            match move:
                case 'op':
                    move_vals[move] = (
                        X / N, 
                        (X ** 2 * (N - X)) / N ** 3
                    )
                case 'saw':
                    move_vals[move] = (
                        2*X / N, 
                        2*(X ** 2 * (N - X)) / N ** 3
                    )
                case 'self':
                    move_vals[move] = (
                        X*(1 - X)/(N*(N - 1)),
                        -X**2*(N - X)*(X - 1)**2*(-N + X + 1)/(N**3*(N - 1)**3)
                    )
                case 'beer':
                    move_vals[move] = (
                        X / N,
                        -X**2*(N - X)*(-N + X + 1)/(N**3*(N - 1))
                    )
                case 'inverter':
                    move_vals[move] = (
                        (N - X)/N,
                        X*(N - X)**2/N**3
                    )
                case 'magnifying_glass':
                    move_vals[move] = (
                        X*(2*N - X - 1)/(N*(N - 1)),
                        X*(N - X)*(N*(N - 1) + X*(-2*N + X + 1))**2/(N**3*(N - 1)**3)
                    )
        
        best_move = 'op'
        best_val = -1
        for move, (mean, variance) in move_vals.items():
            if variance == 0:
                score = mean
            else:
                score = mean / variance
            
            if score > best_val:
                best_move = move
                best_val = score
        
        return best_move
    
    def game_repr(self, board: BuckshotRoulette) -> np.ndarray:
        arr = []
        
        knowledge_map = {True:1, None:0, False:-1}
        for b in self.known_shells:
            arr.append(knowledge_map[b])
        
        if len(arr) < 8:
            arr.extend([0] * (8 - len(arr)))
        
        arr.extend(astuple(board.items[self.me]))
        arr.extend(astuple(board.items[1 - self.me]))
        arr.extend(astuple(board._active_items))
        
        arr.append(board.charges[self.me] / board.max_charges)
        arr.append(board.charges[1 - self.me] / board.max_charges)
        arr.extend(board.shotgun_info())
        arr.append(int(board._skip_next))
        return np.expand_dims(np.array(arr), axis=0)