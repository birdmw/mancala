import pickle
from collections import defaultdict
from datetime import datetime
from os.path import isfile
from random import choice
from time import time

import pandas as pd
from numpy import array as array
from numpy.random import choice as np_choice
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.model_selection import train_test_split


class Board:
    def __init__(self):
        self.pos = {0: 0, 7: 0, "turn": 'player_1'}
        for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]:
            self.pos[i] = 4
        self.history = []

    def play(self, bin, v=0):
        mancala = 7 if "1" in self.pos["turn"] else 0
        opp_mancala = 0 if "1" in self.pos["turn"] else 7

        # if opponent has no stones, sweep lane
        if sum([self.pos[i] for i in range(1, 7)]) == 0:
            stones = sum([self.pos[i] for i in range(8, 14)])
            for i in range(8, 14):
                self.pos[i] = 0
            self.pos[0] += stones
            self.pos['turn'] = None
        if sum([self.pos[i] for i in range(8, 14)]) == 0:
            stones = sum([self.pos[i] for i in range(1, 7)])
            for i in range(1, 7):
                self.pos[i] = 0
            self.pos[7] += stones
            self.pos['turn'] = None

        if bin:

            # pick up stones
            stones = self.pos[bin]
            self.pos[bin] = 0

            # place stones
            for s in range(stones):
                bin += 1
                if bin == 14:
                    bin = 0
                if bin == opp_mancala:
                    bin += 1
                if v:
                    print "adding stone to bin", bin
                self.pos[bin] += 1

            # stealing
            if mancala == 7 and bin in [1, 2, 3, 4, 5, 6] and self.pos[bin] == 1 and self.pos[14 - bin]:
                if v:
                    print "stealing"
                self.pos[bin] = 0
                self.pos[mancala] += 1
                stolen_bin = 14 - bin
                self.pos[mancala] += self.pos[stolen_bin]
                self.pos[stolen_bin] = 0
            elif mancala == 0 and bin in [8, 9, 10, 11, 12, 13] and self.pos[bin] == 1 and self.pos[14 - bin]:
                if v:
                    print "stealing"
                self.pos[bin] = 0
                self.pos[mancala] += 1
                stolen_bin = 14 - bin
                self.pos[mancala] += self.pos[stolen_bin]
                self.pos[stolen_bin] = 0

            # change players
            if bin != mancala:
                if mancala == 7:
                    self.pos["turn"] = "player_2"
                    if v:
                        print "====================================changing players to player 2==========="
                if mancala == 0:
                    self.pos["turn"] = "player_1"
                    if v:
                        print "====================================changing players to player 1==========="
        else:  # if pass take totals
            stones = sum([self.pos[i] for i in range(8, 14)])
            for i in range(8, 14):
                self.pos[i] = 0
            self.pos[0] += stones
            stones = sum([self.pos[i] for i in range(1, 7)])
            for i in range(1, 7):
                self.pos[i] = 0
            self.pos[7] += stones
            self.pos['turn'] = None

        self.history.append(dict(self.pos))


def get_index(csv_name):
    with open('data\\' + csv_name) as f:
        for i, l in enumerate(f):
            pass
    return i


class Dojo:
    def __init__(self, p1, p2, csv_name='raw_data.csv', save=True, v=0):
        self.history = []
        self.board = Board()
        self.p1 = p1
        self.p2 = p2
        self.csv_name = csv_name
        self.save = save
        self.v = v
        if self.save:
            if not isfile('data\\' + csv_name):
                with open('data\\' + csv_name, 'wb') as f:
                    header = ',0,1,2,3,4,5,6,7,8,9,10,11,12,13,turn,winner\n'
                    f.write(header)

    def score(self, v=0):
        if self.board.pos[0] > self.board.pos[7]:
            winner = "player_2"
        elif self.board.pos[0] < self.board.pos[7]:
            winner = "player_1"
        else:
            winner = "tie"

        game_history = list(self.board.history)
        game_history.append(dict(self.board.pos))
        for gh in range(len(game_history)):
            game_history[gh]['winner'] = winner
        chosen = choice(game_history)
        index = get_index(self.csv_name)
        new_line = ''.join(str([index] + [val for k, val in chosen.iteritems()]).split()).strip('[]') + '\n'
        if self.save:
            with open('data\\' + self.csv_name, "a") as myfile:
                myfile.write(new_line)
        if self.v > 1:
            print "winner:", winner
        return winner

    def play_game(self, v=0):
        for i in range(100):
            all_spots = range(1, 7) + range(8, 14)
            # end game if all playable spaces are empty or someone has more than half the points
            if sum([self.board.pos[i] for i in all_spots]) == 0 or self.board.pos[0] > 24 or self.board.pos[7] > 24 or self.board.pos['turn'] == None:
                if v:
                    print "final position:", self.board.pos
                return self.score()

            if self.board.pos['turn'] == "player_1":
                move = self.p1.play(self.board)
            else:
                move = self.p2.play(self.board)
            if v:
                print "------------------------------>", self.board.pos['turn'], "plays bin", move
            self.board.play(move)
            if v:
                print_board(self.board.pos)
            self.history.append(dict(self.board.pos))


def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    try:
        raw = ((val - src[0]) / float(src[1] - src[0])) * float(dst[1] - dst[0]) + dst[0]
    except ZeroDivisionError:
        print "DIVIDE BY ZERO"
        print val
        print src
        print dst
    return raw


def print_board(pos):
    print"======BOARD========"
    print "Left Mancala:", pos[0]
    print [str(pos[i]) for i in range(8, 14)[::-1]]
    print [str(pos[i]) for i in range(1, 7)]
    print "Right Mancala:", pos[7]
    print"=============="


class RandoBot:
    def __init__(self):
        pass

    def play(self, board):
        if board.pos['turn'] == "player_1":
            playable = [1, 2, 3, 4, 5, 6]
        else:
            playable = [8, 9, 10, 11, 12, 13]
        playable = filter(lambda bin: board.pos[bin], playable)
        if playable:
            chosen = choice(playable)
        else:
            return None
        return chosen


class LearnerBot:
    def __init__(self, model_path="data\GBC_model_100.pkl", model=None, monte_carlo=True, time_per_move=.1, v=0):
        self.time_per_move = time_per_move
        if not model:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model = model
        self.monte_carlo = monte_carlo
        self.v = v

    def play(self, board, v=0):
        # print "PLAYING"
        chosen = None
        if v:
            print "playing..."
        if board.pos['turn'] == "player_1":
            playable = [1, 2, 3, 4, 5, 6]
        else:
            playable = [8, 9, 10, 11, 12, 13]
        playable = filter(lambda bin: board.pos[bin], playable)
        if len(playable) == 1:
            return playable[0]
        elif playable:
            odds = []
            for bin in playable:
                b_board = Board()
                b_board.pos = dict(board.pos)
                b_board.play(bin)
                pos = [b_board.pos[i] for i in range(14)]
                if board.pos['turn'] == "player_1":
                    p = [0, 1]
                else:  # board.pos['turn'] == "player_2"
                    p = [1, 0]

                # odds of winning - odds of losing
                pos = array(pos).reshape(1, -1)
                # print self.model.predict_proba(pos)
                my_odds = self.model.predict_proba(pos)[0][p[0]]
                opponents_odds = self.model.predict_proba(pos)[0][p[1]]
                if opponents_odds > 0.0:
                    win_odds = my_odds / opponents_odds
                elif (my_odds - opponents_odds) <= 0.0:
                    win_odds = 0.0
                else:
                    win_odds = my_odds - opponents_odds
                odds.append(win_odds)

            # normalize odds to be from 0 to 1
            min_, max_ = min(odds), max(odds)
            odds = map(lambda x: scale(x, (min_, max_), (0.00, 1.0,)), odds)

            # so their sum == 1
            raw_sum = sum(odds)
            weights = [(float(i) / float(raw_sum)) for i in odds]

            # weighted choice
            if not self.monte_carlo:
                chosen = np_choice(playable, p=weights)

            # TODO: add monte carlo choice
            # Monte Carlo playouts
            if self.monte_carlo:
                # print "monte_carlo", self.v
                end = time() + self.time_per_move
                monte_counts = defaultdict(list)
                mc_count = 0
                while time() < end:
                    mc_count += 1
                    # make a monte_dojo with save=False and p1=LearnerBot(model=self.model, monte_carlo=False) and
                    #                                       p2=LearnerBot(model=self.model, monte_carlo=False)
                    p1 = LearnerBot(model=self.model, monte_carlo=False)
                    p2 = LearnerBot(model=self.model, monte_carlo=False)
                    monte_dojo = Dojo(p1=p1, p2=p2, csv_name="raw_games.csv", save=False)
                    # replace monte_dojo.board.pos with dict(board.pos)
                    monte_dojo.board.pos = dict(board.pos)
                    # choose a move and play it out
                    # print weights
                    chosen = np_choice(playable, p=weights)
                    monte_dojo.board.play(chosen)
                    # have dojo finish the game
                    monte_dojo.play_game()
                    # determine winner with monte_dojo.board.pos[0] and monte_dojo.board.pos[7]
                    if monte_dojo.board.pos[0] < monte_dojo.board.pos[7]:
                        winner = 'player_1'
                    elif monte_dojo.board.pos[0] > monte_dojo.board.pos[7]:
                        winner = 'player_2'
                    else:
                        winner = "tie"
                    # add winner to monte_weights[chosen]
                    monte_counts[chosen].append(winner)
                # player = board.pos['turn'] # which is 'player_1' or 'player_2' or None
                monte_weights = dict(monte_counts)
                player = board.pos['turn']
                opponent = list({'player_1', 'player_2'} - {player})[0]

                for bin, winner_list in monte_weights.iteritems():
                    opponent_wins = float(winner_list.count(opponent))
                    my_wins = float(winner_list.count(player))
                    if opponent_wins:
                        monte_weights[bin] = my_wins / opponent_wins
                    else:
                        monte_weights[bin] = None
                max_odds = max(monte_weights.values())
                for bin, odds in monte_weights.iteritems():
                    if odds == None:
                        monte_weights[bin] = max_odds
                if set(monte_weights.values()) == {None}:
                    for bin, odds in monte_weights.iteritems():
                        if odds == None:
                            monte_weights[bin] = 1.0

                # convert monte_weights from dict(list) to list(float) of wins-losses
                possible_bins = monte_weights.keys()
                weight_list = monte_weights.values()

                # normalize odds to be from 0 to 1
                min_, max_ = min(weight_list), max(weight_list)
                if min_ == max_:
                    min_ = max_ - 1
                odds = map(lambda x: scale(x, (min_, max_), (0.0, 1.0,)), weight_list)

                # so their sum == 1
                raw_sum = sum(odds)
                weights = [(float(i) / float(raw_sum)) for i in odds]

                # chosen = np_choice(playable, p=weights)
                chosen = np_choice(possible_bins, p=weights)
                if self.v:
                    print mc_count, 'game playouts'
                return chosen

        else:
            return None
        if v:
            print chosen
        return chosen


class Human:
    def __init__(self):
        pass

    def play(self, board):
        bin = input("Enter Bin: ")
        print int(bin)
        return int(bin)


def collect_games(p1, p2, csv_name, count=100000, v=0):
    print "collecting games..."
    if isinstance(p1, Human) or isinstance(p2, Human):
        v = 1
    win_count = defaultdict(int)
    session_tagged_history = []
    interest = 1
    for i in range(count):
        if i >= interest:
            interest *= 2
            print len(session_tagged_history), datetime.now()
        dojo = Dojo(p1, p2, csv_name=csv_name, v=1)
        dojo_winner = dojo.play_game(v=v)
        win_count[dojo_winner] += 1
        choice_pos = choice(dojo.history)
        choice_pos['winner'] = dojo_winner
        session_tagged_history.append(dict(choice_pos))

    df = DataFrame(session_tagged_history)


def show_winners(p1, p2):
    win_count = defaultdict(int)
    for i in range(100000):
        if i % 1000 == 0:
            print i
        dojo = Dojo(p1, p2, 'raw_games.csv')
        dojo_winner = dojo.play_game(v=0)
        win_count[dojo_winner] += 1
        print dict(win_count)


def winner(name):
    if name == "'player_1'":
        return 0
    elif name == "'player_2'":
        return 1
    else:  # tie
        return 2


def retrain(data="raw_games.csv", learning_rate=.1, n_estimators=100, max_depth=100):
    df = pd.read_csv("data/" + data)
    # print df.head(10)
    X = df[[str(x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]]
    y = df['winner'].apply(lambda x: winner(x))
    # print y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    try:
        with open("data/GBC_model_500.pkl", 'wb') as f:
            model = pickle.load(f)
            model.set_params(warm_start=True)
            model.fit(X_train, y_train)
    except:
        # print y_train
        model = GBC(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, warm_start=False,
                    verbose=1).fit(X_train,
                                   y_train)

        # print X_train, "example"
    try:
        with open("data/GBC_model_500.pkl", 'wb') as f:
            pickle.dump(model, f)
    except:
        with open("data/GBC_model_500.pkl", 'wb') as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    p1 = RandoBot()
    p2 = p1
    collect_games(p1, p2, csv_name="raw_games.csv", count=100000, v=0)
    retrain("raw_games.csv", learning_rate=.1, n_estimators=200, max_depth=17)
    for i in range(1000):
        try:
             p1 = LearnerBot(model_path="data\GBC_model_500.pkl", time_per_move=(i/10)+2, v=1, monte_carlo=True)
             p2 = p1
        except:
            pass
        #p1 = Human()
        collect_games(p1, p2, csv_name="raw_games.csv", count=100, v=0)
        retrain("raw_games.csv", learning_rate=.1, n_estimators=200, max_depth=17)
