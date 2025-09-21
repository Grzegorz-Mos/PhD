!git clone https://github.com/bcollazo/catanatron.git && cd catanatron && git fetch origin pull/301/head:pr-301 && git checkout pr-301
!pip install -r catanatron/dev-requirements.txt
!pip install -e catanatron/catanatron_core
!pip install -e catanatron/catanatron_server
!pip install -e catanatron/catanatron_gym
!pip install -e catanatron/catanatron_env
!pip install -e catanatron/catanatron_experimental
!git log
exit()

import random
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from catanatron import *
from catanatron.game import *
from catanatron.models.map_instance import *
from catanatron.models.board import Board, longest_acyclic_path
from catanatron.models.enums import *
from catanatron_experimental.play import *



# Initialization of the default map and board, i.e., recommended one in game rules.
DEFAULT_MAP_TILES = initialize_tiles(
  BASE_MAP_TEMPLATE
  ,[ 10 , 9 , 2 , 10 , 12 , 9 , 8 , 5 , 6 , 11 , 5 , 8 , 4 , 6 , 11 , 3 , 4 , 3 ]
  ,[ None , SHEEP , None , None , BRICK , WOOD , None , WHEAT , ORE , None ]
  ,[ None , BRICK , WOOD , SHEEP , ORE , WHEAT , WHEAT , WOOD , BRICK , WHEAT , SHEEP , SHEEP , ORE , SHEEP , BRICK , WOOD , ORE , WHEAT , WOOD , None ]
)

DEFAULT_MAP = MapInstance.from_tiles(DEFAULT_MAP_TILES)
DEFAULT_BOARD = Board( DEFAULT_MAP )



# Redefining Catanatron functions
# There is no possibility of setting own map.
# Thus, it is sufficient to redefine play_batch_core and play_batch.
# This is copy_paste with DEFAULT_MAP set.
def X_play_batch_core(num_games, players, game_config, accumulators=[]):
    for accumulator in accumulators:
        if isinstance(accumulator, SimulationAccumulator):
            accumulator.before_all()

    for _ in range(num_games):
        for player in players:
            player.reset_state()
        catan_map = DEFAULT_MAP #CHANGE
        game = Game(
            players,
            None,
            game_config.discard_limit,
            game_config.vps_to_win,
            catan_map,
        )
        try: #Avoiding invalid simulations due to players restrictions
          game.play(accumulators)
          yield game
        except:
          pass

    for accumulator in accumulators:
        if isinstance(accumulator, SimulationAccumulator):
            accumulator.after_all()

def X_play_batch(
    num_games,
    players,
    output_options=None,
    game_config=None,
    quiet=False,
):
    output_options = output_options or OutputOptions()
    game_config = game_config or GameConfigOptions()

    statistics_accumulator = StatisticsAccumulator()
    vp_accumulator = VpDistributionAccumulator()
    accumulators = [statistics_accumulator, vp_accumulator]
    if output_options.output:
        ensure_dir(output_options.output)
    if output_options.output and output_options.csv:
        accumulators.append(CsvDataAccumulator(output_options.output))
    if output_options.output and output_options.json:
        accumulators.append(JsonDataAccumulator(output_options.output))
    if output_options.db:
        accumulators.append(DatabaseAccumulator())
    for accumulator_class in CUSTOM_ACCUMULATORS:
        accumulators.append(accumulator_class(players=players, game_config=game_config))

    if quiet:
        for _ in X_play_batch_core(num_games, players, game_config, accumulators): #CHANGE
            pass
        return (
            dict(statistics_accumulator.wins),
            dict(statistics_accumulator.results_by_player),
            statistics_accumulator.games,
        )

    # ===== Game Details
    last_n = 10
    actual_last_n = min(last_n, num_games)
    table = Table(title=f"Last {actual_last_n} Games", box=box.MINIMAL)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("SEATING")
    table.add_column("TURNS", justify="right")
    for player in players:
        table.add_column(f"{player.color.value} VP", justify="right")
    table.add_column("WINNER")
    if output_options.db:
        table.add_column("LINK", overflow="fold")

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        CustomTimeRemainingColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task(f"Playing {num_games} games...", total=num_games)
        player_tasks = [
            progress.add_task(
                rich_player_name(player), total=num_games, show_time=False
            )
            for player in players
        ]

        for i, game in enumerate(
            X_play_batch_core(num_games, players, game_config, accumulators) #CHANGE
        ):
            winning_color = game.winning_color()

            if (num_games - last_n) < (i + 1):
                seating = ",".join([rich_color(c) for c in game.state.colors])
                row = [
                    str(i + 1),
                    seating,
                    str(game.state.num_turns),
                ]
                for player in players:  # should be in column order
                    points = get_actual_victory_points(game.state, player.color)
                    row.append(str(points))
                row.append(rich_color(winning_color))
                if output_options.db:
                    row.append(accumulators[-1].link)

                table.add_row(*row)

            progress.update(main_task, advance=1)
            if winning_color is not None:
                winning_index = list(map(lambda p: p.color, players)).index(
                    winning_color
                )
                winner_task = player_tasks[winning_index]
                progress.update(winner_task, advance=1)
        progress.refresh()
    console.print(table)

    # ===== PLAYER SUMMARY
    table = Table(title="Player Summary", box=box.MINIMAL)
    table.add_column("", no_wrap=True)
    table.add_column("WINS", justify="right")
    table.add_column("AVG VP", justify="right")
    table.add_column("AVG SETTLES", justify="right")
    table.add_column("AVG CITIES", justify="right")
    table.add_column("AVG ROAD", justify="right")
    table.add_column("AVG ARMY", justify="right")
    table.add_column("AVG DEV VP", justify="right")
    for player in players:
        vps = statistics_accumulator.results_by_player[player.color]
        avg_vps = sum(vps) / len(vps)
        avg_settlements = vp_accumulator.get_avg_settlements(player.color)
        avg_cities = vp_accumulator.get_avg_cities(player.color)
        avg_largest = vp_accumulator.get_avg_largest(player.color)
        avg_longest = vp_accumulator.get_avg_longest(player.color)
        avg_devvps = vp_accumulator.get_avg_devvps(player.color)
        table.add_row(
            rich_player_name(player),
            str(statistics_accumulator.wins[player.color]),
            f"{avg_vps:.2f}",
            f"{avg_settlements:.2f}",
            f"{avg_cities:.2f}",
            f"{avg_longest:.2f}",
            f"{avg_largest:.2f}",
            f"{avg_devvps:.2f}",
        )
    console.print(table)

    # ===== GAME SUMMARY
    avg_ticks = f"{statistics_accumulator.get_avg_ticks():.2f}"
    avg_turns = f"{statistics_accumulator.get_avg_turns():.2f}"
    avg_duration = formatSecs(statistics_accumulator.get_avg_duration())
    table = Table(box=box.MINIMAL, title="Game Summary")
    table.add_column("AVG TICKS", justify="right")
    table.add_column("AVG TURNS", justify="right")
    table.add_column("AVG DURATION", justify="right")
    table.add_row(avg_ticks, avg_turns, avg_duration)
    console.print(table)

    if output_options.output and output_options.csv:
        console.print(f"GZIP CSVs saved at: [green]{output_options.output}[/green]")

    return (
        dict(statistics_accumulator.wins),
        dict(statistics_accumulator.results_by_player),
        statistics_accumulator.games,
    )



# Functions for transforming edges into sequence
def AreEdgesEqual( edge0 , edge1 ):
  return len( set( edge0 + edge1 ) ) == 2

def AreEdgesNeighbours( edge0 , edge1 ):
  return len( set( edge0 + edge1 ) ) == 3

def IsEdgeDangling( edge , edges ):
  c0 = edge[ 0 ]
  c1 = edge[ 1 ]
  c0counter = 0
  c1counter = 0
  for iEdge in edges:
    ic0 = iEdge[ 0 ]
    ic1 = iEdge[ 1 ]

    if AreEdgesEqual( edge , iEdge ):
      continue

    if c0 == ic0 or c0 == ic1:
      c0counter += 1

    if c1 == ic0 or c1 == ic1:
      c1counter += 1

  if c0counter == 0 or c1counter == 0:
    return True

  return False

def GetDanglingEdge( edges ):
  for edge in edges:
    if IsEdgeDangling( edge , edges ):
      return edge

  return None

def GetLeftNeighbourForEdges( edge0 , edge1 , edges ):
  if AreEdgesEqual( edge0 , edge1 ):
    return None

  for edge in edges:
    if AreEdgesEqual( edge , edge0 ) or AreEdgesEqual( edge , edge1 ):
      continue

    if AreEdgesNeighbours( edge , edge1 ):
      tEdge = tuple(sorted(edge))
      tEdge1 = tuple(sorted(edge1))

      if edges2sequenceItems[ tEdge1 ][ tEdge ] == 0:
        return edge

def GetNumberOfEdgeNeighbours( edge , edges ):
  counter = 0
  for iEdge in edges:
    if AreEdgesNeighbours( edge , iEdge ):
      counter += 1

  return counter

def IsEdgeLeftNeighbour( baseEdge , targetEdge , edgesSequenceDictionary ):
  if not AreEdgesNeighbours( baseEdge , targetEdge ):
    return False

  return edgesSequenceDictionary[ tuple(sorted(baseEdge)) ][ tuple(sorted(targetEdge)) ] == 0

def GetEdgeNeighbours( edge , edges ):
  neighbours = []

  for iEdge in edges:
    if AreEdgesNeighbours( edge , iEdge ):
      neighbours.append( iEdge )

  return neighbours

def GetSequenceFromEdge( edge: tuple , edges: list , edgesSequenceDictionary: dict , areEdgesPrepared = False ):
  if not areEdgesPrepared:
    edges = list(set([ tuple(sorted(e)) for e in edges ]))

  if edge is None:
    edge = GetDanglingEdge( edges )

  if edge is None:
    edge = edges[ 0 ]

  edge = tuple(sorted(edge))

  leftEdges = [ e for e in edges if e != edge ]
  neighbours = GetEdgeNeighbours( edge , edges )
  nNeighbours = len( neighbours )

  if nNeighbours == 0:
    return ( '' , leftEdges )

  firstNeighbours = [ neighbours[0] ] + [ e for e in neighbours if AreEdgesNeighbours( e , neighbours[0] ) ]
  secondNeighbours = [ e for e in neighbours if e not in firstNeighbours ]

  nFirstNeighbours = len( firstNeighbours )
  nSecondNeighbours = len( secondNeighbours )

  if nFirstNeighbours == 1 or nSecondNeighbours == 1:
    if nFirstNeighbours != nSecondNeighbours and nSecondNeighbours == 1:
      firstNeighbours , secondNeighbours = secondNeighbours , firstNeighbours
    neighbour = firstNeighbours[ 0 ]
    c0 = tuple(sorted(edge))
    c1 = tuple(sorted(neighbour))
    sequenceItem = edgesSequenceDictionary[ c0 ][ c1 ]
    return ( sequenceItem + GetSequenceFromEdge( neighbour , leftEdges , edgesSequenceDictionary , True )[0] , leftEdges )

  if nFirstNeighbours == 2 or nSecondNeighbours == 2:
    if nFirstNeighbours < nSecondNeighbours:
      firstNeighbours , secondNeighbours = secondNeighbours , firstNeighbours
    e1 = firstNeighbours[0]
    e2 = firstNeighbours[1]
    c0 = tuple(sorted(edge))
    c1 = tuple(sorted(e1))
    c2 = tuple(sorted(e2))
    d1 = edgesSequenceDictionary[ c0 ][ c1 ]
    d2 = edgesSequenceDictionary[ c0 ][ c2 ]

    if d1 > d2:
      e1 , e2 = e2 , e1

    leftEdges = [ e for e in edges if e not in [ edge , e1 , e2 ] ]
    subSequence1 , subLeftEdges1 = GetSequenceFromEdge( e1 , leftEdges , edgesSequenceDictionary , True )
    subSequence2 , subLeftEdges2 = GetSequenceFromEdge( e2 , subLeftEdges1 , edgesSequenceDictionary , True )

    sequence = '(' + subSequence1 + ')' + subSequence2

    return ( sequence , subLeftEdges2 )

def EdgesToSequencesForFixedEdge( edges , fixedEdge ):
  neighbours = [ e for e in edges if e[0] < e[1] and len( set( fixedEdge + e ) ) == 3 ]
  leftEdges = [ e for e in edges if e[0] < e[1] and len( set( fixedEdge + e ) ) > 2 and e not in neighbours ]

  if len( neighbours ) == 0:
      return [ '' ]

  if len( neighbours ) == 1:
    neighbour = neighbours[0]
    bitValue = edges2sequenceItems[tuple(sorted(fixedEdge))][tuple(sorted(neighbour))]
    return [ '{}{}'.format( bitValue , s ) for s in EdgesToSequencesForFixedEdge( leftEdges , neighbour ) ]

  neighbour0 = neighbours[0]
  neighbour1 = neighbours[1]

  bitValue0 = edges2sequenceItems[tuple(sorted(fixedEdge))][tuple(sorted(neighbour0))]
  bitValue1 = edges2sequenceItems[tuple(sorted(fixedEdge))][tuple(sorted(neighbour1))]

  if bitValue0 == '0':
    leftNeighbour , rightNeighbour = neighbour0 , neighbour1
  else:
    leftNeighbour , rightNeighbour = neighbour1 , neighbour0

  leftPart = EdgesToSequencesForFixedEdge( leftEdges , leftNeighbour )
  rightPart = EdgesToSequencesForFixedEdge( leftEdges , rightNeighbour )

  return [ '({}){}'.format( s1 , s2 ) for s1 in leftPart for s2 in rightPart ]

def splitSequence( seq ):
  if '(' not in seq:
    return [ seq ]

  startBracketPosition = seq.find( '(' )
  startBracketsCounter = 0

  for i in range( startBracketPosition + 1 , len( seq ) ):
    char = seq[ i ]
    if char == '(':
      startBracketsCounter += 1
    if char == ')':
      if startBracketsCounter == 0:
        return [ seq[:startBracketPosition] , seq[startBracketPosition+1:i] , seq[i+1:] ]

      startBracketsCounter -= 1



# Initializing edges direction to sequence items mapper
board = Board(DEFAULT_MAP)

tileEdgesRefs2SequenceItem = {
    EdgeRef.EAST: { EdgeRef.NORTHEAST: '0' , EdgeRef.SOUTHEAST: '1' }
    ,EdgeRef.SOUTHEAST: { EdgeRef.EAST: '0' , EdgeRef.SOUTHWEST : '1' }
    ,EdgeRef.SOUTHWEST: { EdgeRef.SOUTHEAST: '0' , EdgeRef.WEST : '1' }
    ,EdgeRef.WEST: { EdgeRef.SOUTHWEST: '0' , EdgeRef.NORTHWEST : '1' }
    ,EdgeRef.NORTHWEST: { EdgeRef.WEST: '0' , EdgeRef.NORTHEAST : '1' }
    ,EdgeRef.NORTHEAST: { EdgeRef.NORTHWEST: '0' , EdgeRef.EAST : '1' }
}

edges2sequenceItems = {}
tiles = [ tile for tile in board.map.tiles.values() ]

for tile in tiles:
  edges = tile.edges
  for direction , iCoordinates in edges.items():
    for nDirection , inCoordinates in edges.items():
      coordinates = tuple(sorted( iCoordinates ))
      nCoordinates = tuple(sorted( inCoordinates ))

      if len( set( coordinates + nCoordinates ) ) == 3:
        if not ( coordinates in edges2sequenceItems ):
          edges2sequenceItems[ coordinates ] = {}

        edges2sequenceItems[ coordinates ][ nCoordinates ] = tileEdgesRefs2SequenceItem[ direction ][ nDirection ]




# Initializing sequence to edges mapper
sequence2edges = {}
for edge in edges2sequenceItems.keys():
  rEdge = edge[::-1]

  p0 = edge[0]
  p1 = edge[1]

  if not ( edge in sequence2edges ):
    sequence2edges[ edge ] = {}
    sequence2edges[ rEdge ] = {}

  for subedge , item in edges2sequenceItems[ edge ].items():
    sp0 = subedge[0]
    sp1 = subedge[1]

    if sp0 == p1:
      sequence2edges[ edge ][ item ] = subedge

    if sp1 == p1:
      sequence2edges[ edge ][ item ] = subedge[::-1]

    if sp0 == p0:
      sequence2edges[ rEdge ][ item ] = subedge

    if sp1 == p0:
      sequence2edges[ rEdge ][ item ] = subedge[::-1]



# Edges arranger class - getting all possible structure edges combinations
import itertools
import ast

class EdgesArranger:
  def __init__( self , edges:list , fixedEdges:list , arrangedEdges = {} , blockedEdges = {} , deep = 0 ):
    self.edges = [ e for e in edges if e[0] < e[1] ]
    self.fixedEdges = [ tuple( sorted( e ) ) for e in fixedEdges ]
    self.arrangedEdges = { k: arrangedEdges[k] for k in arrangedEdges.keys() }
    self.blockedEdges = { k: blockedEdges[k] for k in blockedEdges.keys() }
    self.deep = deep
    self.neighbourhood = {}

  def isEdgesArrangeBlocked( self , edge1:tuple , edge2:tuple ):
    return ( str(edge1) in self.blockedEdges ) and ( edge2 in self.blockedEdges[ str(edge1) ] )

  def isEdgeArranged( self , edge:tuple ):
    return str(edge) in self.arrangedEdges

  def initEdgeNeighbours( self , edge:tuple ):
    self.neighbourhood[ str( edge ) ] = [ e for e in self.edges if len( set( e + edge ) ) == 3 ]

  def getEdgeNeighbours( self , edge:tuple ):
    edgeStr = str(edge)

    if edgeStr not in self.neighbourhood:
      self.initEdgeNeighbours( edge )

    return self.neighbourhood[ edgeStr ]

  def getBranchedNeighbours( self , edge:tuple ):
    neighbours = self.getEdgeNeighbours( edge )
    return [ ( e1 , e2 ) for e1 in neighbours for e2 in neighbours if len( set( e1 + e2 ) ) == 3 and (e1[0] < e2[0] or e1[1] < e2[1]) ]

  def getAllEdgesCombinations( self , edges ):
    result = [ combination for i in range( len( edges ) ) for combination in itertools.combinations( edges , i + 1 ) ]

    for i in range(len(result)):
      if len(result[i][0]) > 0 and isinstance( result[i][0][0] , tuple ):
        result[i] = result[i][0]

    return result

  def getAllEdgesFamiliesDistinctProducts( self , edgesFamilies ):
    return [ product for product in list( itertools.product( *edgesFamilies ) ) if len( set( product ) ) == len( product ) ]

  def getEdgesNeighbourhoodPossibilites( self , edges ):
    neighbourhoodPossibilites = {}

    for edge in edges:
      neighbourhoodPossibilites[ str(edge) ] =  [((),),]
      neighbourhoodPossibilites[ str(edge) ] += [ ( e , ) for e in self.getEdgeNeighbours( edge ) ]
      neighbourhoodPossibilites[ str(edge) ] += [ bn for bn in self.getBranchedNeighbours( edge ) ]

    return neighbourhoodPossibilites

  def arrange( self ):
    if len( self.edges ) == len( self.arrangedEdges.keys() ):
      return [ self.arrangedEdges ]

    if len( self.fixedEdges ) == 0:
      return []

    fixedEdgesCombinations = self.getAllEdgesCombinations( self.fixedEdges )
    results = []
    for fixedEdgesCombination in fixedEdgesCombinations:
      neighbourhoodPossibilites = self.getEdgesNeighbourhoodPossibilites( fixedEdgesCombination )
      neighbourhoodProducts = self.getAllEdgesFamiliesDistinctProducts( neighbourhoodPossibilites.values() )
      fixedEdgesCombinationLength = len( fixedEdgesCombination )

      for neighbourhoodProduct in neighbourhoodProducts:
        fixedEdges = set()
        arrangedEdges = self.arrangedEdges.copy()
        blockedEdges = self.blockedEdges.copy()
        isArrangeable = True

        for i in range(fixedEdgesCombinationLength):
          if not isArrangeable:
            break

          edge = fixedEdgesCombination[i]
          edgeStr = str(edge)
          neighbours = neighbourhoodProduct[i]

          if edgeStr not in blockedEdges:
            blockedEdges[ edgeStr ] = set()

          if edge == tuple():
            continue

          if len( [ 1 for neighbour in neighbours if edgeStr in blockedEdges and neighbour in blockedEdges[ edgeStr ] ] ) > 0:
            isArrangeable = False
            break

          if len( [ 1 for neighbour in neighbours if str(neighbour) in arrangedEdges ] ) > 0:
            isArrangeable = False
            break

          if len( [ 1 for neighbour in neighbours if neighbour != tuple() and len( [ 1 for vs in arrangedEdges.values() if neighbour in vs ] ) > 0 ] ) > 0:
            isArrangeable = False
            break

          fixedEdges.update( n for n in neighbours )
          arrangedEdges[ edgeStr ] = set( n for n in neighbours )
          for neighbour in neighbours:
            if neighbour == tuple():
              continue
            neighbourStr = str(neighbour)
            if neighbourStr not in blockedEdges:
              blockedEdges[ neighbourStr ] = set()

            blockedEdges[ edgeStr ].update( [ n for n in self.getEdgeNeighbours(edge) if len( set( neighbour + n ) ) == 4 and n not in neighbours and n != tuple() ] )
            blockedEdges[ neighbourStr ].update( [ e for e in self.getEdgeNeighbours(neighbour) if len( set( edge + e ) ) <= 3 ] )

        if isArrangeable:
          for e in self.fixedEdges:
            if len(e) > 0 and str(e) not in arrangedEdges.keys():
              arrangedEdges[ str(e) ] = set(((),))
          results += EdgesArranger( self.edges , fixedEdges , arrangedEdges , blockedEdges , self.deep + 1 ).arrange()

    return [ast.literal_eval(el1) for el1 in set([str(el2) for el2 in results])]

# EXAMPLE: Edges arranger class
tEdges = [(2, 3), (2, 9), (3, 4), (3, 12), (4, 15), (10, 11), (11, 12), (14, 15), (14, 37)]
arranges = EdgesArranger(tEdges,[(14,37)]).arrange()
arranges



# Edges arrange to sequences
from ast import literal_eval as make_tuple
def ArrangeToSequence( arrange , fixedEdge , edgesSequenceDictionary ):
  key = str(tuple(sorted(fixedEdge)))
  values = arrange[ key ]

  if values == set(((),)):
    return ''

  iterValues = iter(values)
  value0 = next(iterValues)
  bit0 = edgesSequenceDictionary[ make_tuple(key) ][ value0 ]

  if len( values ) == 1:
    return bit0 + ArrangeToSequence( arrange , value0 , edgesSequenceDictionary )

  value1 = next(iterValues)
  if bit0 == '1':
    value0 , value1 = value1 , value0

  return '({}){}'.format( ArrangeToSequence( arrange , value0 , edgesSequenceDictionary ) , ArrangeToSequence( arrange , value1 , edgesSequenceDictionary ) )

# EXAMPLE: Arranging edges and converting to sequences
edges = [(13, 14), (38, 39), (40, 42), (17, 39), (18, 40), (37, 38), (15, 17), (41, 42), (14, 15), (40, 44), (16, 21), (16, 18), (17, 18), (14, 37), (36, 37)]
a = EdgesArranger( edges , [( 13 , 14 )] ).arrange()
seqs = set( ArrangeToSequence( ae , (13,14) , edges2sequenceItems ) for ae in a )
print( seqs )



# Get all polygonal chains in sequence
def sequence2polygonalChainSequences( sequence ):
  splited = splitSequence( sequence )

  if len( splited ) == 1:
    return splited

  result = []
  result += [ '{}0{}'.format( splited[0] , rest ) for rest in sequence2polygonalChainSequences( splited[1] ) ]
  result += [ '{}1{}'.format( splited[0] , rest ) for rest in sequence2polygonalChainSequences( splited[2] ) ]

  return result



# Function to display board with weighted edges
from operator import le
from IPython.display import display, HTML
sqrt3 = 1.73205080757

#Reference: https://www.redblobgames.com/grids/hexagons/
def CubeToPixels( coordinate ):
  q , s , r = coordinate

  x = sqrt3 * ( q  + r / 2 )
  y = 3 * r / 2

  return ( x , y )

#catanatron/ui/src/utils/coordinates.js
def GetEdgeTransform( direction: EdgeRef , size ):
  distanceToEdge = size * 0.865

  edgeRef2deg = {
      EdgeRef.NORTHEAST: 30
      ,EdgeRef.EAST: 90
      ,EdgeRef.SOUTHEAST: 150
      ,EdgeRef.SOUTHWEST: 210
      ,EdgeRef.WEST: 270
      ,EdgeRef.NORTHWEST : 330
  }

  return f'translateX(-50%) translateY(-50%) rotate({edgeRef2deg[ direction ]}deg) translateY(-{distanceToEdge}px)'

  match direction:
    case EdgeRef.NORTHEAST:
      return format.format(30)
    case EdgeRef.EAST:
      return format.format(90)
    case EdgeRef.SOUTHEAST:
      return format.format(150)
    case EdgeRef.SOUTHWEST:
      return format.format(210)
    case EdgeRef.WEST:
      return format.format(270)
    case EdgeRef.NORTHWEST:
      return format.format(330)

def GetHtmlToDisplayBoardWithWeightedEdges( weightedEdges , board = DEFAULT_BOARD , mapWidth = 500 , mapHeight = 500 ):
  htmlCatanMap = ''
  a = mapHeight / 10 #Length of the triangle side
  h = a * sqrt3 / 2 #Height of the triangle

  for coordinate , tile in [ ( c , t ) for ( c , t ) in board.map.tiles.items() if isinstance( t , LandTile ) ]:
    x , y = CubeToPixels( coordinate )
    left = mapWidth * ( x / 10 + 4 / 10 )
    top = mapHeight * ( y / 10 + 4 / 10 )
    resource = tile.resource


    if resource == None:
      resource = 'desert'
    else:
      resource = resource.lower()

    htmlCatanMap += f'<div style="left: {left}px; top: {top}px; height: calc( {a}px * 2 ); width: calc( {h}px * 2 );" class="tile {resource}"></div>'

    edges = tile.edges
    for edgeRef , edgeId in edges.items():
      sortedEdgeId = tuple(sorted(edgeId))
      id = '_'.join( str(i) for i in sortedEdgeId )
      if htmlCatanMap.find(f'id="e_{id}"') == -1:
        transform = GetEdgeTransform( edgeRef , mapHeight / 10 )
        divClass = 'edge'
        style = ''
        style += f'left: {left+h}px;'
        style += f'top: {top+a}px;'
        style += f'transform: {transform};'
        style += f'width: {a}px;'
        if sortedEdgeId in weightedEdges:
          style += f'opacity: {weightedEdges[ sortedEdgeId ]};'
          divClass += ' highlighted'
        htmlCatanMap += f'<div id="e_{id}" class="{divClass}" style="{style}"></div>'

  htmlCatanMap = f'<div class="map-container" style="height: {mapHeight}px; width: {mapWidth}px;">{htmlCatanMap}</div>'

  return htmlCatanMap

def TransformSequenceToEdgesForEdge( sequence: str , fixedDirectedEdge: tuple ):
    edges = [ fixedDirectedEdge ]
    sequenceParts = splitSequence( sequence )

    edge = fixedDirectedEdge
    for item in sequenceParts[0]:
      nextEdge = sequence2edges[ edge ][ item ]

      edges.append( nextEdge )

      if edge[1] == nextEdge[0]:
        edge = nextEdge
      else:
        edge = nextEdge[::-1]

    if len( sequenceParts ) > 1:
      lEdge = sequence2edges[ edge ][ '0' ]
      rEdge = sequence2edges[ edge ][ '1' ]
      edges += TransformSequenceToEdgesForEdge( sequenceParts[1] , lEdge )
      edges += TransformSequenceToEdgesForEdge( sequenceParts[2] , rEdge )

    return edges

def TransformSequencesToEdgesForEdge( sequences: list , fixedDirectedEdge: tuple ):
  edgesList = []

  for sequence in sequences:
    edgesList.append( TransformSequenceToEdgesForEdge( sequence , fixedDirectedEdge ) )

  return edgesList

def EdgesList2WeightedEdges( edgesList: list = [] ):
  n = len( edgesList )

  weightedEdges = {}
  for edges in edgesList:
    for edge in edges:
      e = tuple(sorted(edge))

      if not ( e in weightedEdges ):
        weightedEdges[ e ] = 0

      weightedEdges[ e ] += 1

  return { e: c / n for ( e , c ) in weightedEdges.items() }

def GetHtmlToDisplayBoardFromSequencesFromEdge( sequences = [] , fixedDirectedEdge = () , board = DEFAULT_BOARD , mapWidth = 500 , mapHeight = 500 ):
  edgesList = TransformSequencesToEdgesForEdge( sequences , fixedDirectedEdge )
  weightedEdges = EdgesList2WeightedEdges( edgesList )

  return GetHtmlToDisplayBoardWithWeightedEdges( weightedEdges , board , mapWidth , mapHeight )

def DisplayMapHtmlBodies( htmlBodies = [] ):
  htmlStyle = ''
  for resource in [ 'brick' , 'desert' , 'ore' , 'sheep' , 'wheat' , 'wood' ]:
    htmlStyle += f'.{resource} {{ background-image: url( "./files/content/catanatron/ui/src/assets/tile_{resource}.svg" ) }}'
  htmlStyle += '.tile { background-repeat: no-repeat; background-size: 100% 100%; position: absolute; display: flex;}'
  htmlStyle += '.edge { position: absolute; display: flex; border-radius: 10px; z-index: 1;}'
  htmlStyle += '.highlighted { border: 2px solid red; background-color: red; }'
  htmlStyle += '.map-container { position: relative; border: 2px solid black; display: inline-block; }'

  htmlStyle = f'<style>{htmlStyle}</style>'
  htmlHead = htmlStyle

  htmlBody = ''.join( htmlBodies )

  html = f'<html><head>{htmlHead}</head><body>{htmlBody}</body></html>'
  display(HTML(html))



# Function to extract data from games
def extractDataFromGames( games , targetEdge ):
  results = []
  c0 = targetEdge[0]
  c1 = targetEdge[1]

  n = len( games )
  for i in range( n ):
    game = games[ i ]
    board = game.state.board
    firstPlayer = game.state.colors[0]
    connected_component = None
    color = None
    player = None

    for _color in board.connected_components:
      if len( board.connected_components[ _color ] ) != 1:
        continue

      _connected_component = board.connected_components[ _color ][0]

      if c0 in _connected_component and c1 in _connected_component:
        connected_component = _connected_component
        color = _color
        player = 'P{}'.format( game.state.colors.index( color ) )
        break

    if color != game.winning_color():
      continue

    if connected_component == None:
      continue

    edges = list( set( tuple(sorted(e)) for ( e , c ) in board.roads.items() if c == color ) )
    edges.sort(key = lambda e: "{:02d}".format(e[0]) + ';' + "{:02d}".format(e[1]) )

    if not IsEdgeDangling( targetEdge , edges ):
      continue

    if len( [ e for e in edges if c0 in e and c1 not in e ] ) > 0:
      continue

    arranges = EdgesArranger(edges,[tuple(sorted(targetEdge))]).arrange()
    sequences = set( ArrangeToSequence( ae , targetEdge , edges2sequenceItems ) for ae in arranges )

    if len( sequences ) > 1:
      continue

    fPolygonalChains = [ sequence2polygonalChainSequences( s ) for s in sequences ]

    lenMaxPolygonalChain = max( [ max( [ len( pc ) for pc in pcs ] ) for pcs in fPolygonalChains ] )

    polygonalChains = [ pc for pcs in fPolygonalChains for pc in pcs if len( pc ) == lenMaxPolygonalChain ]

    dataItem = {
        'sequences': sequences
        ,'polygonalChains': polygonalChains
        ,'edges': edges
        ,'isWinner': color == game.winning_color()
        ,'player': player
        ,'color': color.value
        ,'victoryPoints': game.state.player_state[ '{}_VICTORY_POINTS'.format(player) ]
        ,'gameTurns': int(game.state.num_turns)
        ,'gameId': game.id
    }

    results.append( dataItem )

  return results



# Running and extracting games
%%capture
targetEdge = ( 49 , 22 )
class FixedPlayer(Player):
    def decide(self, game, playable_actions):
      #Starting from the fixed edge
      if game.state.colors[0] == self.color:
        fixedEdge = targetEdge
        fixedFirstAction = Action( self.color , ActionType.BUILD_ROAD , fixedEdge )
        if is_valid_action( game.state , fixedFirstAction ):
          return fixedFirstAction

        #Avoiding building some edges preserving the direction from the fixed edge
        pa = [ a for a in playable_actions if a.action_type != ActionType.BUILD_ROAD or len( set( a.value + fixedEdge ) ) == 4 ]
        return random.choice(pa)

      return random.choice(playable_actions)

players = [
    FixedPlayer(Color.RED),
    FixedPlayer(Color.BLUE),
    FixedPlayer(Color.WHITE),
    FixedPlayer(Color.ORANGE),
]

results = []
numberOfGames = 10000
while 1:
  wins, results_by_player, games = X_play_batch( 100 , players , None , GameConfigOptions( 7 , 10 ) )
  results += extractDataFromGames( games , targetEdge)
  if len( results ) >= 1000:
    break



# Exporting result
import json
def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_sets_to_lists(item) for item in obj]
    return obj


from google.colab import drive
drive.mount('/content/drive')
save_path = '/content/drive/My Drive/simulationResult.json'

# Save the object to Google Drive
with open(save_path, 'w') as file:
    json.dump(convert_sets_to_lists( results ), file, indent=4)



# Result
sequences = [ HexSeq( item[ 'sequences' ][ 0 ] ) for item in results ]

params = [
    [ extendedHammingDistance , 'Extended Hamming Distance' , 1 ]
    ,[ extendedWeightedHammingDistance , 'Extended Weighted Hamming Distance' , 1 ]
    ,[ bsLevenshteinDistance , 'Levebshtein Distance' , 1 ]
]

for param in params:
  distanceFunction = param[ 0 ]
  distanceTitle = param[ 1 ]
  distancePower = param[ 2 ]

  medoid = getMedoid( sequences , distanceFunction , distancePower )
  print( distanceTitle , "\n" , medoid[ 'value' ] , medoid[ 'distance' ] , sum( medoid[ 'distances' ] ) )
  DisplayMapHtmlBodies( GetHtmlToDisplayBoardFromSequencesFromEdge( [ str( medoid[ 'value' ] ) ] , targetEdge ) )
