# ===============================================================================================
#  Fractional programming example using PulP for LP Solve 5.5 documentation.  For details, see:
# 
#     http://lpsolve.sourceforge.net/5.5/ratio.htm
#
#  The idea is to find the state of N binary decision variables: {x1, x2, ..., xN} that optimizes
#  the following objective function (which is the ratio of two linear functions), given the linear
#  constraints.
#  
#
#     max     n0 + n1 * x1 + n2 * x2 + ... + nN * xN
#            ----------------------------------------
#             d0 + d1 * x1 + d2 * x2 + ... + dN * xN
#     
#     s.t.  c1_1 x1 + c1_2 x2 + ... + c1_N x2 <= b_1
#           c2_1 x1 + c2_2 x2 + ... + c2_N xN <= b_2
#           ...
#           cK_1 x1 + cK_2 x2 + ... + cK_N xN <= b_K
#           x1, x2 in {0, 1}
#
#
#  INSTALLATION prerequisites:
#  --------------------------------------
#    conda create -n lin-prog python=3.6
#    source activate lin-prog
#    conda install -n lin-prog ipython
#    conda install -n lin-prog numpy
#    conda install -n lin-prog lpsolve
#    conda install -n lin-prog glpk
#    pip install PulP
# ===============================================================================================


import pulp

# -----------------------------------------------------------------------------------------------
#  Part 1: Solve the linear program with continuous variables.  These results will be used to
#          determine constraints for the binary problem in Part 2.
# -----------------------------------------------------------------------------------------------

# Instantiate our problem class
model = pulp.LpProblem("lp solve fractional 0-1 programming example", pulp.LpMaximize)

y0 = pulp.LpVariable('y0', lowBound=0, cat='Continuous') # 'Continuous', 'Integer', 'Binary'
y1 = pulp.LpVariable('y1', lowBound=0, cat='Continuous')
y2 = pulp.LpVariable('y2', lowBound=0, cat='Continuous')

model += 1.8 * y1 + 1.7 * y2

model +=  -6 * y0 + 1.5 * y1 +       y2 <= 0
model += -20 * y0 + 3.0 * y1 +   4 * y2 <= 0
model +=  10 * y0 + 4.0 * y1 + 4.1 * y2 == 1

model.solve()
pulp.LpStatus[model.status]

print(f'x1 value: {y1.varValue / y0.varValue}')         # 1.3333333439111112
print(f'x2 value: {y2.varValue / y0.varValue}')         # 4.0
print(f'maximum value: {pulp.value(model.objective)}')  # 0.28991596659999996
print(f'solution time: {model.solutionTime}')


# -----------------------------------------------------------------------------------------------
#  Part 2: Use the results from Part 1 to impose additional constraints that allow the problem
#          to be solved with binary decision variables (xi, for i >= 1 in this example).
# -----------------------------------------------------------------------------------------------
#
# PROBLEM: 
#
#     max         1.8 x1 + 1.7 x2  
#           ----------------------
#            10 + 4.0 x1 + 4.1 x2
#     
#     s.t.  1.5 x1 +   x2 <= 6
#           3.0 x1 + 4 x2 <= 20
#           x1, x2 in {0, 1} (binary)

import pulp

model = pulp.LpProblem("lp solve fractional 0-1 programming example", pulp.LpMaximize)

y0 = pulp.LpVariable('y0', lowBound=0, cat='Continuous') # 'Continuous', 'Integer', 'Binary'
y1 = pulp.LpVariable('y1', lowBound=0, cat='Continuous')
y2 = pulp.LpVariable('y2', lowBound=0, cat='Continuous')

# Auxiliary binary variables needed to create fractional 0/1 linear programming problem
# via Charnes-Cooper (1962) transformation.
# (https://en.wikipedia.org/wiki/Linear-fractional_programming).
z1 = pulp.LpVariable('z1', cat='Binary')
z2 = pulp.LpVariable('z2', cat='Binary')

model += 1.8 * y1 + 1.7 * y2

# Modified objective denominator after Charnes-Cooper transformation.
model +=  10 * y0 + 4.0 * y1 + 4.1 * y2 == 1

# Orginal constraints after Charnes-Cooper transformation.
model +=  -6 * y0 + 1.5 * y1 +       y2 <= 0
model += -20 * y0 + 3.0 * y1 +   4 * y2 <= 0

# Additional constraints to make y_i binary values, for i >= 1.
# Using f1 = 0.127.  f1 = 10 in example. yi <= 0.12605042, for i >= 1.
# Using f2 = 0.032.  f2 = 10 in example. y0  = 0.031512605.
model +=  y1 <= 0.127 * z1
model +=  y1 - y0 - 0.032 * z1 >= -0.032
model +=  y1 - y0 + 0.032 * z1 <= 0.032

model +=  y2 <= 0.127 * z2
model +=  y2 - y0 - 0.032 * z2 >= -0.032
model +=  y2 - y0 + 0.032 * z2 <= 0.032

model.solve()
pulp.LpStatus[model.status]

print(f'x1 value: {y1.varValue / y0.varValue}')         # 1.0
print(f'x2 value: {y2.varValue / y0.varValue}')         # 1.0
print(f'maximum value: {pulp.value(model.objective)}')  # 0.1933701665
print(f'solution time: {model.solutionTime}')


# -----------------------------------------------------------------------------------------------
#  Part 3: Check random values returned by optimization versus brute force search.
#          Compare the wall clock performance versus some different problem sizes.
# -----------------------------------------------------------------------------------------------

