import sys
def getMedoid( sequences: list , distanceFunction , powDist = 1 , weights = None ):
  n = len( sequences )
  if weights == None:
    weights = [ 1 for _ in range( n ) ]

  weightsSum = sum( weights )
  distances = [ None for _ in range( n ) ]
  _distances = [ 0 for _ in range( n ) ]
  result = None
  minDistance = sys.maxsize

  for i in range( n ):
    tempDistance = 0
    for j in range( n ):
      _distance = weights[j] * ( distanceFunction( sequences[i] , sequences[j] ) ** powDist )
      _distances[j] = _distance
      tempDistance += _distance

    tempDistance /= weightsSum
    if tempDistance < minDistance:
      minDistance = tempDistance
      distances = _distances.copy()
      result = sequences[i]

  return { 'value': result , 'distance': minDistance , 'sequences': sequences , 'weights': weights , 'distances': distances }
