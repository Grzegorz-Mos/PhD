def tauStr( binarySequenceString: str ) -> str:
	return ''.join('1' if b == '0' else '0' for b in binarySequenceString)

@numba.njit
def tauInt( binarySequenceInteger: int , length: int ) -> int:
    mask = (1 << length) - 1
    return binarySequenceInteger ^ mask

def sigmaStr( binarySequenceString: str ) -> str:
  return binarySequenceString[::-1]

@numba.njit
def sigmaInt(bsInt: int, length: int) -> int:
    result = 0
    for _ in range(length):
        result = (result << 1) | (bsInt & 1)
        bsInt >>= 1
    return result

def piStr( binarySequenceString: str ) -> str:
  return sigmaStr( tauStr( binarySequenceString ) )

@numba.njit
def piInt( binarySequenceInteger: int , length: int ) -> int:
  return sigmaInt( tauInt( binarySequenceInteger , length ) , length )

