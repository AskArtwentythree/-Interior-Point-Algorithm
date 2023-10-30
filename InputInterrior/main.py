import numpy
import numpy as np

def input(coeff_main,n,tableau,coeff_constr,approximation_accuracy):
    print("Enter the coefficients of the main problem on one line with space delimiter")
    coeff_main = list(map(int, input().split()))
    print("Enter number of constraints")
    n = int(input())
    tableau = []
    for i in range(n):
        tableau.append(list(map(int, input("Enter coefficients of " + str(i + 1) + "th constraint with space delimiter")
                                .split(' '))))
    print("Enter the right-hand coefficients of the constraints on one line with space delimiter")
    coeff_constr = list(map(int, input().split()))
    coeff_constr.append(0)
    print("Enter 1 if your problem is maximize and -1 if your problem is minimize")
    k = int(input())
    for i in range(len(coeff_main)):
        coeff_main[i] = coeff_main[i] * k
    for i in range(len(coeff_main)):
        coeff_main[i] = coeff_main[i] * -1
    print("Enter the approximation accuracy")
    approximation_accuracy = int(input())
    tableau.append(coeff_main.copy())
    for i in range(0, len(tableau)):
        tableau[i].append(coeff_constr[i])