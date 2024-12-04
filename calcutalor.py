import polynoms as pn
import numpy as np
from scipy.integrate import odeint
class Calculator:
    def __init__(self, startValues, maxValues, polynomCoefs, disturbancesCoefs, normValues):
        self.values = startValues
        self.maxValues = maxValues
        self.functions = self.initFunctions(polynomCoefs)
        self.qfunctions = self.initFunctions(disturbancesCoefs)
        self.normValues = normValues

    def initFunctions(self, coefs):
        functions = []
        for coef_set in coefs:
            num_coefs = len(coef_set)

            if num_coefs == 2:
                polynomial = pn.LinearPolynomial(*coef_set)
            elif num_coefs == 3:
                polynomial = pn.QuadraticPolynomial(*coef_set)
            elif num_coefs == 4:
                polynomial = pn.CubicPolynomial(*coef_set)
            else:
                raise ValueError(f"Неподдерживаемое количество коэффициентов: {num_coefs}")

            functions.append(polynomial)

        return functions


    def calculate(self, timeIntervals):
        solution = odeint(self.calcFunctions, self.values, timeIntervals)

        return solution

    def calcFunctions(self, u, t):
        [L1_t, L2_t, L3_t, L4_t, L5_t, L6_t, L7_t, L8_t, L9_t, L10_t, L11_t, L12_t, L13_t, L14_t, L15_t] = u

        q1 = self.qfunctions[0].calc(t)
        q2 = self.qfunctions[1].calc(t)
        q3 = self.qfunctions[2].calc(t)
        q4 = self.qfunctions[3].calc(t)

        dL1_dx = -(self.functions[0].calc(L10_t) * self.functions[1].calc(L11_t) * self.functions[2].calc(L14_t))

        dL2_dx = self.functions[3].calc(L3_t) * self.functions[4].calc(L7_t) * self.functions[5].calc(L8_t)* \
                 self.functions[6].calc(L9_t) * self.functions[7].calc(L13_t) - (self.functions[8].calc(L10_t) *
                 self.functions[9].calc(L11_t) * self.functions[10].calc(L14_t) * self.functions[11].calc(L15_t) * q1 + q2 + q3 + q4)

        dL3_dx = self.functions[12].calc(L1_t) - (self.functions[13].calc(L15_t) * q1 + q3 + q4)

        dL4_dx = self.functions[14].calc(L1_t)

        dL5_dx = self.functions[15].calc(L1_t) * q2 - q1

        dL6_dx = q2 - (self.functions[16].calc(L4_t) * self.functions[17].calc(L11_t) * self.functions[18].calc(L12_t) *
                       self.functions[19].calc(L14_t) * q1)

        dL7_dx = self.functions[20].calc(L5_t) * self.functions[21].calc(L6_t) * self.functions[22].calc(L13_t) * \
                 self.functions[23].calc(L15_t) * q1 + q2 + q3

        dL8_dx = self.functions[24].calc(L5_t) * self.functions[25].calc(L6_t) * self.functions[26].calc(L11_t) * \
                 self.functions[27].calc(L13_t) * self.functions[28].calc(L14_t) * self.functions[29].calc(
            L15_t) * q1 + q2 + q3

        dL9_dx = self.functions[30].calc(L3_t) * self.functions[31].calc(L13_t) * q2 - (
                    self.functions[32].calc(L10_t) * self.functions[33].calc(L11_t) * self.functions[34].calc(
                L14_t) * q1)

        dL10_dx = self.functions[35].calc(L3_t) * self.functions[36].calc(L9_t) * self.functions[37].calc(
            L15_t) * q1 + q2 + q3 + q4

        dL11_dx = self.functions[38].calc(L3_t) * self.functions[39].calc(L13_t) * self.functions[40].calc(
            L14_t) * q1 + q3 - (self.functions[41].calc(L15_t) * q4)

        dL12_dx = self.functions[42].calc(L11_t) * self.functions[43].calc(L13_t) * self.functions[44].calc(
            L14_t) * q1 + q2 + q3 - (self.functions[45].calc(L15_t))

        dL13_dx = self.functions[46].calc(L2_t) * self.functions[47].calc(L3_t) * q2

        dL14_dx = self.functions[48].calc(L11_t) * self.functions[49].calc(L12_t) * self.functions[50].calc(
            L13_t) * q1 + q2

        dL15_dx = self.functions[51].calc(L2_t) * self.functions[52].calc(L3_t) * self.functions[53].calc(L13_t) * \
                  self.functions[54].calc(L14_t) * q1 + q2

        return [dL1_dx, dL2_dx, dL3_dx, dL4_dx, dL5_dx, dL6_dx, dL7_dx, dL8_dx, dL9_dx, dL10_dx, dL11_dx, dL12_dx, dL13_dx, dL14_dx, dL15_dx]

