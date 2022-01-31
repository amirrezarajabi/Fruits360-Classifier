import copy

import numpy as np
from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """

        # TODO (Additional: Learning curve)
        self.learning_curve(players)
        return self.choosen_players(method="roulette-wheel")(players, num_players)
        

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            new_parents = [self.clone_player(player) for player in prev_players]
            new_players = []
            ps = self.choosen_players("Q-tournament")(new_parents, num_players)
            for i in range(0, num_players, 2):
                p1 = ps[i]
                p2 = ps[i + 1]
                children = self.crossover(p1, p2, alpha=0.7, crossover_method="cal-crossover")
                self.mutate(children[0], 0.1)
                self.mutate(children[1], 0.1)
                new_players.append(children[0])
                new_players.append(children[1])
            return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def top_k(self, players, num_players):
        players_ = sorted(players, key=lambda x: x.fitness, reverse=True)
        return players_[: num_players]

    def roulette_wheel(self, players, num_players):
        population_fitnes = sum([player.fitness for player in players])
        player_probabilities = [player.fitness / population_fitnes for player in players]
        return np.random.choice(players, num_players, p=player_probabilities)
    
    def roulette_wheel_mapped(self, players, num_players):
        pf = [player.fitness for player in players]
        max_ = max(pf)
        mean_ = sum(pf) / len(pf)
        population_fitnes = sum([player.fitness for player in players])
        player_probabilities = [(1.5 / num_players - 1 / len(pf)) * (player.fitness - mean_) / (max_ - mean_) + 1 / len(pf) for player in players]
        return np.random.choice(players, num_players, p=player_probabilities)

    
    def SUS(self, players, num_players):
        population_fitnes = sum([player.fitness for player in players])
        player_probabilities = []
        index = 0
        for player in players:
            player_probabilities.append((player.fitness / population_fitnes, index))
            index += 1
        player_probabilities.sort(key=lambda x: x[0], reverse=True)
        index = 0
        for i in range(len(player_probabilities)):
            player_probabilities[i] = (player_probabilities[i][0], player_probabilities[i][1] , index, player_probabilities[i][0] + index)
            index += player_probabilities[i][0]
        choose_from = []
        sus_uniform = float(np.random.uniform(0, 1/num_players))
        j = 0
        for i in range(num_players):
            while j < len(player_probabilities) and not (player_probabilities[j][2] <= sus_uniform <= player_probabilities[j][3]):
                j += 1
            choose_from.append(players[player_probabilities[j][1]])
            sus_uniform += 1/num_players
        return choose_from

    def Q_tournament(self, players, num_players):
        Q = 15
        choosen_players = []
        for _ in range(num_players):
            players_to_choose = np.random.choice(players, Q, replace=False)
            choosen_players.append(max(players_to_choose, key=lambda x: x.fitness))
        return choosen_players
    
    def choosen_players(self, method="Q-tournament"):
        if method == "Q-tournament":
            return self.Q_tournament
        elif method == "SUS":
            return self.SUS
        elif method == "roulette-wheel":
            return self.roulette_wheel
        elif method == "top-k":
            return self.top_k
        elif method == "roulette-wheel-mapped":
            return self.roulette_wheel_mapped
        else:
            raise Exception("Method not implemented")

    def learning_curve(self, players):
        mean_ = 0
        max_  = players[0].fitness
        min_  = players[0].fitness
        for player in players:
            mean_ += player.fitness
            if player.fitness > max_:
                max_ = player.fitness
            if player.fitness < min_:
                min_ = player.fitness
        mean_ /= len(players)
        data = f"{mean_},{max_},{min_}\n"
        with open("json.rj", "a") as f:
            f.write(data)
        f.close()
    
    def two_points_crossover(self, player1, player2):
        """
        Gets two players as an input and produces a child.
        """
        child1 = Player(self.game_mode)
        child2 = Player(self.game_mode)
        child1.nn = copy.deepcopy(player1.nn)
        child2.nn = copy.deepcopy(player2.nn)
        for i in range(len(player1.nn.W)):
            shape_w, size_w = player1.nn.W[i].shape, player1.nn.W[i].size
            child1.nn.W[i].flatten()[size_w // 3:2*size_w // 3] = child2.nn.W[i].flatten()[size_w // 3:2*size_w // 3]
            child2.nn.W[i].flatten()[size_w // 3:2*size_w // 3] = child1.nn.W[i].flatten()[size_w // 3:2*size_w // 3]
            child1.nn.W[i].reshape(shape_w)
            child2.nn.W[i].reshape(shape_w)
            shape_b, size_b = player1.nn.B[i].shape, player1.nn.B[i].size
            child1.nn.B[i].flatten()[size_b // 3:2*size_b // 3] = child2.nn.B[i].flatten()[size_b // 3:2*size_b // 3]
            child2.nn.B[i].flatten()[size_b // 3:2*size_b // 3] = child1.nn.B[i].flatten()[size_b // 3:2*size_b // 3]
            child1.nn.B[i].reshape(shape_b)
            child2.nn.B[i].reshape(shape_b)
        return child1, child2

    def mutate(self, player, threshold=0.1):
        """
        Gets a player as an input and produces a mutated player.
        """
        for k in range(len(player.nn.W)):
            for i in range(player.nn.W[k].shape[0]):
                for j in range(player.nn.W[k].shape[1]):
                    if np.random.uniform(0, 1) < threshold:
                        player.nn.W[k][i, j] += np.random.normal(0, 1) * 0.3 * player.nn.W[k][i, j]
                if(np.random.uniform(0, 1) < threshold):
                    player.nn.B[k][i, 0] += np.random.normal(0, 1) * 0.3 * player.nn.B[k][i, 0]

    def cal_crossover(self, player1, player2, alpha=0.75):
        child1 = Player(self.game_mode)
        child2 = Player(self.game_mode)
        child1.nn = copy.deepcopy(player1.nn)
        child2.nn = copy.deepcopy(player2.nn)
        for i in range(len(player1.nn.W)):
            child1.nn.W[i] = alpha * player1.nn.W[i] + (1 - alpha) * player2.nn.W[i]
            child2.nn.W[i] = alpha * player2.nn.W[i] + (1 - alpha) * player1.nn.W[i]
            child1.nn.B[i] = alpha * player1.nn.B[i] + (1 - alpha) * player2.nn.B[i]
            child2.nn.B[i] = alpha * player2.nn.B[i] + (1 - alpha) * player1.nn.B[i]
        return child1, child2
    
    def crossover(self, player1, player2, alpha=0.6, crossover_method="two-points", threshold = 0.8):
        """
        Gets two players as an input and produces a child.
        """
        if(np.random.uniform(0, 1, 1) < threshold):
            if crossover_method == "two-points":
                children = self.two_points_crossover(player1, player2)
                return children
            elif crossover_method == "cal-crossover":
                children = self.cal_crossover(player1, player2, alpha)
                return children
            else:
                raise ValueError("Crossover model not supported")
        else:
            return player1, player2
            