import decimal

class Input:
    def __init__(self):
        self.C = []
        self.A = []
        self.b = []
        self.approximation = 0
       
    def processInput(self):
        print("Input in format:\n" +
              "First -> Enter coefficients of the main problem \n C[1] ... C[n]\n" +
              "Second -> Enter coefficients of 'i' th constraint with space delimiter\n A[1,1] ... A[1,n]\n" +
              "        ...\n" +
              " A[m,1] ... A[m,n]\n" +
              "Third -> Enter the right-hand coefficients of the constraints on one line with space delimiter\n ->  b[1] ... b[m]\n" +
              "Fourth ->Enter the approximation\n")
    
        C = input().split(" ")
        self.C = [float(x) for x in C]

        AList = []
        while True:
            line = input().split(" ")
            line = [float(x) for x in line]
            if len(line) == 1:
                self.approximation = int(line[0])
                self.b = AList.pop()
                break
            AList.append(line)
        self.A = AList

    def getC(self):
        return self.C

    def getA(self):
        return self.A

    def getB(self):
        return self.b

    def getApproximation(self):
        return self.approximation

class Matrix:
        
    def subtractMatrices(matrix1, matrix2):
        numRows = len(matrix1)
        numCols = len(matrix1[0])
        result = [[0.0] * numCols for _ in range(numRows)]
        for i in range(numRows):
            for j in range(numCols):
                result[i][j] = matrix1[i][j] - matrix2[i][j]
        return result
    
    def subtract(matrix1, matrix2):
        n = len(matrix1)
        result = [0.0] * n
        for i in range(n):
            result[i] = matrix1[i] - matrix2[i]
        return result
    
    def transpose(matrix):
        numRows = len(matrix)
        numCols = len(matrix[0])
        result = [[0.0] * numRows for _ in range(numCols)]
        for i in range(numRows):
            for j in range(numCols):
                result[j][i] = matrix[i][j]
        return result
    
    def inverse(matrix):
        n = len(matrix)
        if n != len(matrix[0]):
            raise ValueError("Matrix must be square for inversion.")
        augmentedMatrix = [[0.0] * (2 * n) for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                augmentedMatrix[i][j] = matrix[i][j]
                augmentedMatrix[i][j + n] = 1.0 if i == j else 0.0
        
        for col in range(n):
            pivotRow = col
            for i in range(col + 1, n):
                if abs(augmentedMatrix[i][col]) > abs(augmentedMatrix[pivotRow][col]):
                    pivotRow = i
            
            temp = augmentedMatrix[col]
            augmentedMatrix[col] = augmentedMatrix[pivotRow]
            augmentedMatrix[pivotRow] = temp
            
            pivotValue = augmentedMatrix[col][col]
            for j in range(2 * n):
                #print(int(pivotValue))
                augmentedMatrix[col][j] /= pivotValue
            
            for i in range(n):
                if i != col:
                    factor = augmentedMatrix[i][col]
                    for j in range(2 * n):
                        augmentedMatrix[i][j] -= factor * augmentedMatrix[col][j]
        
        inverseMatrix = [[0] * n for _ in range(n)]
        for i in range(n):
            inverseMatrix[i] = augmentedMatrix[i][n:]
        return inverseMatrix
 
    def multiplyMatrices(matrix1, matrix2):
        numRows1 = len(matrix1)
        numCols1 = len(matrix1[0])
        numRows2 = len(matrix2)
        numCols2 = len(matrix2[0])
        if numCols1 != numRows2:
            raise ValueError("Matrix dimensions are incompatible for multiplication.")
        result = [[0] * numCols2 for _ in range(numRows1)]
        for i in range(numRows1):
            for j in range(numCols2):
                for k in range(numCols1):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        return result
 
    def matrixOptimalCompare(matrix1, matrix2, differenceAccuracy):
        numRows = len(matrix1)
        numCols = len(matrix1[0])
        for i in range(numRows):
            for j in range(numCols):
                if abs(matrix1[i][j] - matrix2[i][j]) > differenceAccuracy:
                    return False
        return True
 
    def multiply(matrix, array):
        m = len(matrix)
        n = len(array)
        result = [0] * m
        for i in range(m):
            for j in range(n):
                result[i] += matrix[i][j] * array[j]
        return result
 
    def multiplyScalar(array, scalar):
        n = len(array)
        result = [0] * n
        for i in range(n):
            result[i] = array[i] * scalar
        return result
    
    def add(a, b):
        n = len(a)
        result = [0] * n
        for i in range(n):
            result[i] = a[i] + b[i]
        return result
 
    def findMin(array):
        minVal = array[0]
        for i in range(1, len(array)):
            if array[i] < minVal:
                minVal = array[i]
        return minVal
 
    def norm(array, ord):
        sumVal = 0
        for value in array:
            sumVal += abs(value) ** ord
        return sumVal ** (1.0 / ord)

class InteriorPoint:
    def __init__(self):
        self.x = []

    def solve(self, inp, output):
        print("Input the initial feasible solution in format:\n" +
              "x[1] ... x[2 * n]")
        x_input = input().split(" ")
        self.x = [float(x) for x in x_input]
        print("Interior method with alpha 0.5:")
        output.outputResult(self.solves(inp.getC(), inp.getA(), inp.getB(), 0.5), inp.getApproximation())
        print("\n")
        self.x = [float(x) for x in x_input]
        print("Interior method with alpha 0.9:")
        output.outputResult(self.solves(inp.getC(), inp.getA(), inp.getB(), 0.9), inp.getApproximation())

    def solves(self, con, a, B, al):
            csz = len(con)
            asz = len(a)
            C = [0.0] * (csz + asz)
            alpha = al
            for i in range(csz + asz):
                if i < csz:
                    C[i] = con[i]
                else:
                    C[i] = 0
            A = [[0.0] * (csz + asz) for _ in range(asz)]
            for i in range(asz):
                std = 1
                if B[i] < 0:
                    std = -1
                    B[i] *= -1
                for j in range(asz + csz):
                    A[i][j] = a[i][j] if j < csz else 0 if j - csz != i else 1
                    A[i][j] *= std
            while True:
                v = self.x.copy()
                D = [[0] * len(self.x) for _ in range(len(self.x))]
                for j in range(len(self.x)):
                    D[j][j] = self.x[j]
                AA = Matrix.multiplyMatrices(A, D)
                cc = Matrix.multiply(D, C)
                I = [[0] * len(self.x) for _ in range(len(self.x))]
                for j in range(len(self.x)):
                    I[j][j] = 1.0
                F = Matrix.multiplyMatrices(AA, Matrix.transpose(AA))
                FI = Matrix.inverse(F)
                H = Matrix.multiplyMatrices(Matrix.transpose(AA), FI)
                P = Matrix.subtractMatrices(I, Matrix.multiplyMatrices(H, AA))
                cp = Matrix.multiply(P, cc)
                nu = abs(Matrix.findMin(cp))
                one = [1.0] * len(self.x)
                y = Matrix.add(one, Matrix.multiplyScalar(cp, alpha / nu))
                yy = Matrix.multiply(D, y)
                self.x = yy.copy()
                if Matrix.norm(Matrix.subtract(yy, v), 2) < 0.00001:
                    break
            ans = 0
            for j in range(csz):
                ans += self.x[j] * C[j]
            return (ans, self.x[:csz])
        
class Output:
    def outputResult(self, result, approximation):
        if result[0] == float('-inf'):
            print("The method is not applicable!")
            return
        if min:
            print("The maximum of the function is " + str(decimal.Decimal(result[0]).quantize(decimal.Decimal('0.' + '0' * approximation), rounding=decimal.ROUND_HALF_DOWN)))
        print("The vector x* is ", end="")
        for i in range(len(result[1])):
            print(str(decimal.Decimal(result[1][i]).quantize(decimal.Decimal('0.' + '0' * approximation), rounding=decimal.ROUND_HALF_DOWN)) + " ", end="")

input_obj = Input()
out = Output()
input_obj.processInput()

inpt = InteriorPoint()
inpt.solve(input_obj, out)