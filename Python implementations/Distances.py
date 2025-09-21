# Implementation of Extended Hamming Distance for binary sequences
def bsExtendedHammingDistance( s1:HexSeq , s2:HexSeq ):
    str1 , str2 = s1.prefix , s2.prefix
    n1 , n2 = len( str1 ) , len( str2 )
    m , M = min( n1 , n2 ) , max( n1 , n2 )

    distance = sum( [ int( str1[ i ] != str2[ i ] ) for i in range( m ) ] ) + sum( [ 1 for i in range( m , M ) ] )

    return distance

# Implementation of Extended Weighted Hamming Distance for binary sequences
def bsExtendedWeightedHammingDistance( s1:HexSeq , s2:HexSeq ):
    str1 , str2 = s1.prefix , s2.prefix
    n1 , n2 = len( str1 ) , len( str2 )
    m , M = min( n1 , n2 ) , max( n1 , n2 )

    return sum( [ int( str1[ i ] != str2[ i ] ) * (  2 ** ( -i ) ) for i in range( m ) ] ) + sum( [ (  2 ** ( -i ) ) for i in range( m , M ) ] )

# Implementation of Levenshtein Distance for binary sequences
def bsLevenshteinDistance(s1: HexSeq, s2: HexSeq) -> int:
    str1 , str2 = s1.prefix , s2.prefix
    n1, n2 = len(str1), len(str2)
    if n1 < n2:
        str1, str2, n1, n2 = str2, str1, n2, n1

    dp = list(range(n2 + 1))
    for i in range(1, n1 + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n2 + 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j-1] + 1, prev + (str1[i-1] != str2[j-1]))

    return dp[n2]

# Implementation of distance for extended binary sequences using distance for binary sequences
def ebsDistance( s1: HexSeq , s2: HexSeq , bsDistance ):
  isEBS1 , isEBS2 = s1.isExtendedBinarySequence() , s2.isExtendedBinarySequence()

  if not isEBS1 and not isEBS2:
    return bsDistance( s1 , s2 )

  if not isEBS1:
    return ebsDistance( s2 , s1 , bsDistance )

  subSequences1 = s1.getSubSequences()
  if not isEBS2:
    return min( ebsDistance( subSequences1[ 0 ] , s2 , bsDistance ) + ebsDistance( subSequences1[ 1 ] , HexSeq('') , bsDistance ),
               ebsDistance( subSequences1[ 1 ] , s2 , bsDistance ) + ebsDistance( subSequences1[ 0 ] , HexSeq('') , bsDistance ) )

  subSequences2 = s2.getSubSequences()
  return min(
      sum( ebsDistance( subSequences1[ i ] , subSequences2[ i ] , bsDistance ) for i in range( 2 ) ),
      sum( ebsDistance( subSequences1[ i ] , subSequences2[ 1 - i ] , bsDistance ) for i in range( 2 ) ) )


def extendedHammingDistance( s1:HexSeq , s2:HexSeq ):
  return ebsDistance( s1 , s2 , bsExtendedHammingDistance )

def extendedWeightedHammingDistance( s1:HexSeq , s2:HexSeq ):
  return ebsDistance( s1 , s2 , bsExtendedWeightedHammingDistance )

def LevenshteinDistance( s1:HexSeq , s2:HexSeq ):
  return ebsDistance( s1 , s2 , bsLevenshteinDistance )

# Examples
pairsToComparise = [
  ( HexSeq( '()(())()' ) , HexSeq( '101010' ) ),
  ( HexSeq( '0001101010' ) , HexSeq( '1010111' ) ),
  ( HexSeq( '()' ) , HexSeq( '101010' ) ),
  ( HexSeq( '()(())()' ) , HexSeq( '11(00)11' ) )
]

for pairToComparise in pairsToComparise:
  s1 = pairToComparise[ 0 ]
  s2 = pairToComparise[ 1 ]
  print( s1 , '&' , s2 , '=>' , extendedHammingDistance( s1 , s2 ) , '&' , extendedWeightedHammingDistance( s1 , s2 ) , '&' , LevenshteinDistance( s1 , s2 ) )
