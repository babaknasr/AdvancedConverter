import numpy as np
import math
from collections import defaultdict

# Constants
K2C = 273.15 #C

class ConverterClass(object):
    '''
    Conversion Class
    '''
    def __init__(self) -> None:
        '''
        Define conversion list
        convList (dict):
            (unit1, unit2): {'factor': XX, 'add': XX, 'ignore':False/True, func: self.XX, func_inv: self.XX}
        '''
        self.convList = {
            # Non-dimensional
            ('nd', None): {'ignore': True}, # ignore conversion
            ('-', None): {'ignore': True}, # ignore conversion
            ('', None): {'ignore': True}, # ignore conversion
            # Temperature
            ('C', 'K'): {'factor': 1, 'add': K2C},
            ('F', 'K'): {'factor': 5/9, 'add': K2C-(32*5/9)},
            ('R', 'K'): {'factor': 5/9},
            # relative temperature (temperature difference)
            ('K', 'K (rel)'): {'ignore': True},
            ('C', 'K (rel)'): {'factor': 1},
            ('F', 'K (rel)'): {'factor': 5/9},
            ('R', 'K (rel)'): {'factor': 5/9},
            # Length
            ('in', 'm'): {'factor': 0.0254},
            ('ft', 'm'): {'factor': 0.3048},
            #Area
            ('in2', 'm2'): {'factor': 0.0254**2},
            ('ft2', 'm2'): {'factor': 0.3048**2},
            # Volume
            ('in3', 'm3'): {'factor': 0.0254**3},
            ('ft3', 'm3'): {'factor': 0.3048**3},
            # Mass
            ('lb', 'kg'): {'factor': 1/2.204622621848776},
            # Density
            ('lb/ft3', 'kg/m3'): {'factor': 1/0.06242796057614462},
            # Specific Volume
            ('ft3/lb', 'm3/kg'): {'factor': 1/16.0184635218},
            # Velocity
            ('fpm', 'm/s'): {'factor': 1/196.8503937007874},
            # Volumetric flow rate
            ('L/min', 'm3/s'): {'factor': 1/60000},
            ('gpm', 'm3/s'): {'factor': 1/15850.32223705108},
            ('cfm', 'm3/s'): {'factor': 1/2118.880003289315},
            # Mass flow rate
            ('lb/hr', 'kg/s'): {'factor': 1/7936.6414386556},
            # Pressure
            ('kPa', 'Pa'): {'factor': 1000},
            ('psi', 'Pa'): {'factor': 1/0.0001450377377302092},
            ('inH2O', 'Pa'): {'factor': 1/0.00401865},
            ('mmH2O', 'Pa'): {'factor': 1/0.10197162129779},
            # Heat
            ('kJ', 'J'): {'factor': 1000},
            ('kJ/kg', 'J/kg'): {'factor': 1000},
            ('Btu/lb', 'J/kg'): {'factor': 1/0.0004299226137871357},
            ('kW', 'W'): {'factor': 1000},
            ('Btu/hr', 'W'): {'factor': 1/3.4121416351331},
            ('Ton', 'W'): {'factor': 1/0.0002843451},
            # Entropy
            ('kJ/kg-K', 'J/(kg.K)'): {'factor': 1000},
            ('Btu/lb-F', 'J/(kg.K)'): {'factor': 1/0.000238845896627},
            # To be Continue....

            # Custom converter functions...
            ('area', 'volume'): {'func': self.func_A2V, 'func_inv': self.func_V2A}, # just for sample
            
            }

        self.convList = self.PrepareList(self.convList)

    def PrepareList(self, convList):
        convListExtended = convList.copy()
        for key, value in convList.items():
            if (key[1], key[0]) not in convList:
                if value.get('func') is not None and value.get('func_inv') is not None: # if custom func presents
                    convListExtended[(key[1], key[0])] = {'func': value['func_inv'], 'func_inv': value['func'], 'ignore': value.get('ignore', False)}
                elif value.get('ignore', False):    # if ignore is True
                    convListExtended[(key[1], key[0])] = {'ignore': value.get('ignore', False)}
                else:
                    convListExtended[(key[1], key[0])] = {'factor': 1/value['factor'], 'add': -value.get('add', 0)/value['factor'], 'ignore': value.get('ignore', False)}
        return convListExtended

    def Calc(self, value, unit1, unit2, extra=list()):
        value, unit1, unit2, extra = self.check(value, unit1, unit2, extra)
        if self.convList.get((unit1, unit2)) is not None:
            converted_value = self._calc(value, unit1, unit2, extra)
        else:
            converted_value = self.sequence_calc(value, unit1, unit2, extra)
        return converted_value

    def check(self, value, unit1, unit2, extra):
        if unit1 == unit2:
            self.convList[(unit1, unit2)] = {'factor':1, 'ignore': True}
        if not isinstance(extra, list):
            extra = [extra]
        return value, unit1, unit2, extra

    def _calc(self, val, unit1, unit2, extra=list()):
        # Bypass if 'ignore' is True
        if self.convList[(unit1, unit2)].get('ignore', False):
            conv_val = val
            return conv_val
        if self.convList[(unit1, unit2)].get('func'): # incase custom function is defined
            conv_val = self.convList[(unit1, unit2)]['func'](np.array(val), extra)
        else:   # in case of factor, addon
            conv_val = np.array(val)*self.convList[(unit1, unit2)]['factor'] + self.convList[(unit1, unit2)].get('add', 0)
        # Check the value is 0-dim np.array for single value return or more for list value return
        if conv_val.ndim == 0:  # 0-dimension numpy array
            conv_val = float(conv_val)
        else:
            conv_val = list(conv_val)

        return conv_val

    def sequence_calc(self, val, unit1, unit2, extra=list()):
        exist1 = 0
        exist2 = 0
        for t in self.convList.keys():
            if unit1 in t:
                exist1 = 1
            if unit2 in t:
                exist2 = 1
            if exist1*exist2:
                break
        if exist1*exist2:
            short_path = self.BFS_SP(self.connected_graph(list(self.convList.keys())), unit1, unit2)
            if len(short_path) > 1:
                seq = [(short_path[i], short_path[i+1]) for i in range(len(short_path)-1)]
                conv_val = val
                for t in seq:
                    conv_val = self._calc(conv_val, t[0], t[1], extra)
        else:
            conv_val = None
        return conv_val

    def connected_graph(self, edges):
        '''
        Create a list of connected edges
        '''
        # edges = list(self.convList.keys())
        graph = defaultdict(list)        
        
        for edge in edges:    
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])
        return graph

    def BFS_SP(self, graph, start, target):
        '''
        Breath-First Search (BFS) for the Shortest Path
        '''
        # trivial condition
        if start == target:
            SP = [start]
            return new_path

        explored = []
        SP = None
        # queue for traversing the graph in the BFS
        queue = [[start]]
        # Loop to traverse the graph using queue
        while queue:
            path = queue.pop(0)
            node = path[-1]
            # check if the current node is not visited
            if node not in explored:
                neighbors = graph[node]
                # iterate over the neighbors of the node
                for neighbor in neighbors:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                    # Check if the neighbor node is the target
                    if neighbor == target:
                        SP = new_path
                        return SP
                explored.append(node)
    
        return SP

#######################################
# --- Custom Convertion functions --- #
    def func_A2V(self, val, extra):
        return val*extra[0]

    def func_V2A(self, val, extra):
        return val/extra[0]
