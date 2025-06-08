import random

from Classes.Constants import *
from Classes.Hand import Hand
from Classes.Materials import Materials
from Classes.TradeOffer import TradeOffer
from Interfaces.AgentInterface import AgentInterface


class AlexElenaMarcosGeneticAgent(AgentInterface):
    _current_chromosome = None

    def __init__(self, agent_id):
        super().__init__(agent_id)
        # Utilitza el cromosoma actual de la classe
        # Default es mejor cromosoma registrado
        self.chromosome = self.__class__._current_chromosome or [
            92, 133, 67, 152, 132, 159,57, 185, 176, 121, 152, 171
        ]

        # Mapeig de gens a noms
        self.gene_map = {
            'build_road': 0,
            'build_town': 1,
            'build_city': 2,
            'buy_card': 3,
            'thief_aggression': 4,
            'pref_cereal': 5,
            'pref_mineral': 6,
            'pref_clay': 7,
            'pref_wood': 8,
            'pref_wool': 9,
            'accept_trades': 10,
            'make_trades': 11
        }

    @classmethod
    def with_chromosome(cls, chromosome):
        """Configura el cromosoma per a les properes instàncies"""
        cls._current_chromosome = chromosome
        return cls

    def get_gene(self, gene_name):
        return self.chromosome[self.gene_map[gene_name]]

    def set_gene(self, gene_name, value):
        self.chromosome[self.gene_map[gene_name]] = value

    # Funció auxiliar per trobar l'índex del valor màxim en una llista
    def _argmax(self, values):
        max_value = max(values)
        return values.index(max_value)

    # Funció auxiliar per trobar l'índex del valor mínim en una llista
    def _argmin(self, values):
        min_value = min(values)
        return values.index(min_value)

    def on_game_start(self, board_instance):
        """Selecció inicial de poble i carretera"""
        self.board = board_instance

        # Estrategia: Seleccionar nodes amb millors recursos
        valid_nodes = self.board.valid_starting_nodes()
        if not valid_nodes:
            return 0, 0

        # Valorar nodes per recursos adjacents
        node_scores = []
        for node_id in valid_nodes:
            score = 0
            terrains = self.board.nodes[node_id]['contacting_terrain']
            for t_id in terrains:
                terrain = self.board.terrain[t_id]
                resource_type = terrain['terrain_type']

                # Utilitzar preferències del cromosoma
                if resource_type == TerrainConstants.CEREAL:
                    score += self.get_gene('pref_cereal')
                elif resource_type == TerrainConstants.MINERAL:
                    score += self.get_gene('pref_mineral')
                elif resource_type == TerrainConstants.CLAY:
                    score += self.get_gene('pref_clay')
                elif resource_type == TerrainConstants.WOOD:
                    score += self.get_gene('pref_wood')
                elif resource_type == TerrainConstants.WOOL:
                    score += self.get_gene('pref_wool')

            node_scores.append(score)

        # Seleccionar el millor node
        best_node = valid_nodes[self._argmax(node_scores)]
        possible_roads = self.board.nodes[best_node]['adjacent']

        # Seleccionar carretera amb més opcions d'expansió
        if possible_roads:
            return best_node, random.choice(possible_roads)
        return best_node, best_node + 1  # Fallback

    def on_turn_start(self):
        """Decidir si jugar carta de desenvolupament a l'inici del torn"""
        # Si tenim carta de cavaller i hi ha lladre, jugar-la
        knight_cards = self.development_cards_hand.find_card_by_effect(
            DevelopmentCardConstants.KNIGHT_EFFECT)

        if knight_cards:
            return self.development_cards_hand.select_card(0)
        return None

    def on_commerce_phase(self):
        """Generar ofertes de comerç"""
        if random.random() < self.get_gene('make_trades'):
            # Convertir Materials a llista per processar
            resources_list = list(self.hand.resources)

            # Identificar recurs més abundant i més necessitat
            abundant = self._argmax(resources_list)
            needed = self._argmin(resources_list)

            return TradeOffer(
                gives=Materials.from_ids(abundant, 1),
                receives=Materials.from_ids(needed, 1)
            )
        return None

    def on_trade_offer(self, board_instance, offer=TradeOffer(), player_id=int):
        """Respondre a ofertes de comerç"""
        # Acceptar si l'oferta és favorable
        if offer.gives.has_more(offer.receives) and random.random() < self.get_gene('accept_trades'):
            return True
        return False

    def on_build_phase(self, board_instance):
        """Decidir què construir durant la fase de construcció"""
        build_options = []

        # Comprovar si podem construir carretera
        if self.hand.resources.has_more(BuildConstants.ROAD):
            build_options.append(('road', self.get_gene('build_road')))

        # Comprovar si podem construir poble
        if self.hand.resources.has_more(BuildConstants.TOWN):
            build_options.append(('town', self.get_gene('build_town')))

        # Comprovar si podem construir ciutat
        if self.hand.resources.has_more(BuildConstants.CITY):
            build_options.append(('city', self.get_gene('build_city')))

        # Comprovar si podem comprar carta
        if self.hand.resources.has_more(BuildConstants.CARD):
            build_options.append(('card', self.get_gene('buy_card')))

        if not build_options:
            return None

        # Seleccionar acció basada en probabilitats del cromosoma
        actions, probs = zip(*build_options)
        total_prob = sum(probs)
        normalized_probs = [p / total_prob for p in probs]

        # Selecció amb probabilitats
        rand_val = random.random()
        cumulative = 0
        for i, prob in enumerate(normalized_probs):
            cumulative += prob
            if rand_val <= cumulative:
                action = actions[i]
                break
        else:
            action = actions[-1]  # Per seguretat

        # Processar l'acció seleccionada
        if action == 'road':
            valid_roads = self.board.valid_road_nodes(self.id)
            if valid_roads:
                road = random.choice(valid_roads)
                return {
                    'building': BuildConstants.ROAD,
                    'node_id': road['starting_node'],
                    'road_to': road['finishing_node']
                }

        elif action == 'town':
            valid_towns = self.board.valid_town_nodes(self.id)
            if valid_towns:
                return {
                    'building': BuildConstants.TOWN,
                    'node_id': random.choice(valid_towns),
                    'road_to': None
                }

        elif action == 'city':
            valid_cities = self.board.valid_city_nodes(self.id)
            if valid_cities:
                return {
                    'building': BuildConstants.CITY,
                    'node_id': random.choice(valid_cities),
                    'road_to': None
                }

        elif action == 'card':
            return {'building': BuildConstants.CARD, 'node_id': None, 'road_to': None}

        return None

    def on_moving_thief(self):
        """Moure el lladre i seleccionar jugador a robar"""
        # Seleccionar terreny amb més recursos valuosos
        terrain_scores = []
        for terrain in self.board.terrain:
            if terrain['has_thief']:
                continue

            score = 0
            resource_type = terrain['terrain_type']
            if resource_type == TerrainConstants.CEREAL:
                score = self.get_gene('pref_cereal')
            elif resource_type == TerrainConstants.MINERAL:
                score = self.get_gene('pref_mineral')
            elif resource_type == TerrainConstants.CLAY:
                score = self.get_gene('pref_clay')
            elif resource_type == TerrainConstants.WOOD:
                score = self.get_gene('pref_wood')
            elif resource_type == TerrainConstants.WOOL:
                score = self.get_gene('pref_wool')

            terrain_scores.append(score)

        # Seleccionar el millor terreny
        if terrain_scores:
            terrain_id = self._argmax(terrain_scores)
        else:
            terrain_id = random.randint(0, 18)

        # Seleccionar jugador per robar (agressivitat del cromosoma)
        players = [p for p in range(4) if p != self.id]

        player_to_rob = random.choice(players)

        return {'terrain': terrain_id, 'player': player_to_rob}

    def on_having_more_than_7_materials_when_thief_is_called(self):
        """Descartar recursos menys valuosos"""
        # Creem una còpia de la mà actual
        hand_copy = Hand()
        hand_copy.resources = Materials(*self.hand.resources)

        # Obtenim les preferències de recursos
        prefs = {
            MaterialConstants.CEREAL: self.get_gene('pref_cereal'),
            MaterialConstants.MINERAL: self.get_gene('pref_mineral'),
            MaterialConstants.CLAY: self.get_gene('pref_clay'),
            MaterialConstants.WOOD: self.get_gene('pref_wood'),
            MaterialConstants.WOOL: self.get_gene('pref_wool')
        }

        # Ordenem recursos de menys a més valuós
        materials_sorted = sorted(prefs.keys(), key=lambda m: prefs[m])

        # Descartem fins a tenir 7 o menys recursos
        while hand_copy.get_total() > 7:
            for material in materials_sorted:
                if hand_copy.get_from_id(material) > 0:
                    hand_copy.remove_material(material, 1)
                    break

        return hand_copy

    def on_turn_end(self):
        """Decidir si jugar carta al final del torn"""
        # Jugar cartes de victòria si ens donen la victòria
        development_cards = self.development_cards_hand.hand

        for i, card in enumerate(development_cards):
            if card.type == DevelopmentCardConstants.VICTORY_POINT:
                return self.development_cards_hand.select_card(i)

        return None

    def on_monopoly_card_use(self):
        """Seleccionar recurs més valuós per a monopolitzar"""
        # Preferències del cromosoma
        prefs = {
            0: self.get_gene('pref_cereal'),
            1: self.get_gene('pref_mineral'),
            2: self.get_gene('pref_clay'),
            3: self.get_gene('pref_wood'),
            4: self.get_gene('pref_wool')
        }

        # Seleccionar recurs amb més preferència
        return max(prefs.keys(), key=lambda m: prefs[m])

    def on_road_building_card_use(self):
        """Seleccionar carreteres per construir"""
        valid_roads = self.board.valid_road_nodes(self.id)

        if len(valid_roads) >= 2:
            return {
                'node_id': valid_roads[0]['starting_node'],
                'road_to': valid_roads[0]['finishing_node'],
                'node_id_2': valid_roads[1]['starting_node'],
                'road_to_2': valid_roads[1]['finishing_node']
            }
        elif len(valid_roads) == 1:
            return {
                'node_id': valid_roads[0]['starting_node'],
                'road_to': valid_roads[0]['finishing_node'],
                'node_id_2': None,
                'road_to_2': None
            }
        return None

    def on_year_of_plenty_card_use(self):
        """Seleccionar recursos més necessaris"""
        # Convertir Materials a llista per processar
        resources_list = list(self.hand.resources)

        # Seleccionar dos recursos més escassos
        material1 = self._argmin(resources_list)
        resources_list[material1] += 1  # Simular que n'hem afegit un

        material2 = self._argmin(resources_list)

        return {'material': material1, 'material_2': material2}