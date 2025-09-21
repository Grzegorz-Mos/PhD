class HexSeq:
  def __init__( self , extendedBinarySequenceString ):
    if not '(' in extendedBinarySequenceString:
      self.prefix = extendedBinarySequenceString
      self.infix = None
      self.sufix = None
    else:
      bracketsCounter = 1
      i = extendedBinarySequenceString.index( '(' )
      n = len( extendedBinarySequenceString )
      for j in range( i + 1 , n ):
        if extendedBinarySequenceString[ j ] == '(':
          bracketsCounter += 1
        elif extendedBinarySequenceString[ j ] == ')':
          bracketsCounter -= 1

        if bracketsCounter == 0:
          break

      self.prefix = extendedBinarySequenceString[:i]
      self.infix = extendedBinarySequenceString[i+1:j]
      self.sufix = extendedBinarySequenceString[j+1:]

  def getPrefix( self ):
    return HexSeq( self.prefix )

  def isExtendedBinarySequence( self ):
    return self.infix != None

  def getFirstSubSequence( self ):
    return HexSeq( f"{self.prefix}0{self.infix}" )

  def getSecondSubSequence( self ):
    return HexSeq( f"{self.prefix}1{self.sufix}" )

  def getSubSequences( self ):
    if not self.isExtendedBinarySequence():
      return [ HexSeq( self.prefix ) ]

    return [ self.getFirstSubSequence() , self.getSecondSubSequence() ]

  def compare( self , ebs ):
    if (not self.isExtendedBinarySequence()) and (not ebs.isExtendedBinarySequence()):
      if self.prefix >= ebs.prefix:
        return 1
      if self.prefix <= ebs.prefix:
        return -1
      return 0

    if (not self.isExtendedBinarySequence()) and ebs.isExtendedBinarySequence():
      if self.isNotLower( ebs.getFirstSubSequence() ) or self.isNotLower( ebs.getSecondSubSequence() ):
        return -1
      return 0

    if self.isExtendedBinarySequence() and ebs.isExtendedBinarySequence():
      c1 = self.getFirstSubSequence().compare( ebs.getFirstSubSequence() )
      c2 = self.getSecondSubSequence().compare( ebs.getSecondSubSequence() )
      print( c1 , c2 )
      if c1 < 0 and c2 < 0:
        return -1
      if c1 > 0 and c2 > 0:
        return 1
      return 0

  def __repr__(self):
    if self.infix == None:
      return self.prefix

    return f"{self.prefix}({self.infix}){self.sufix}"
