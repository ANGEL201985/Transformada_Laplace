from sympy import symbols, Function, laplace_transform, exp, inverse_laplace_transform, Eq, solve, diff, simplify, Heaviside

# Definir variables
t,s = symbols('t,s', real=True) #  t, s: Variables simbólicas para tiempo y dominio de Laplace.
y = Function('y')(t) #  Define y como función de t (ej. y(t)).
Y = symbols('Y') # Símbolo para Y(s) (transformada de Laplace de y(t)).

# Formando mi ecuacion diferencial
ecuacion_diferencial = Eq(diff(y,t,t) - 2*diff(y,t)+ y, exp(t))

L_y = laplace_transform(y, t, s, noconds=True) # L_y: Calcula la Transformada de Laplace de y(t).
L_dy = s*L_y - y.subs(t,0) # transformada de la primera deriva L(y') = sY - y(0)
L_2dy = s**2*L_y - s*y.subs(t,0)- diff(y,t).subs(t,0) # transformada de la segunda derivada L(y'') = s²Y - sy(0) - y'(0)
L_exp  = laplace_transform(exp(t),t,s, noconds=True) # Transformada de exponencial L(e^t) = 1/(s-1)

# Construir ecuación transformada con la funcion Eq: L(y'' - 2y' + y) = L(e^t)
ecuacion_transformada = Eq(L_2dy - 2*L_dy + L_y, L_exp)

# Sustituir condiciones iniciales y(0)=1, y'(0)=0
ecuacion_transformada = ecuacion_transformada.subs({y.subs(t,0):1, diff(y,t).subs(t,0):0})

# Resolver para Y(s), Despeja algebraicamente Y(s) de la ecuación transformada.
solucion_Y = solve(ecuacion_transformada, L_y)

# Calcular transformada inversa
solucion_final = inverse_laplace_transform(solucion_Y[0],s,t).subs(Heaviside(t),1)
solucion_simplificada = simplify(solucion_final) # Simplifica la expresión final.

print(solucion_simplificada)