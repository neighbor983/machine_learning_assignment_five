import random
from math import exp

DataSet = [
    [ 0.0, 0.0 ],
    [ 1.0, 0.0 ],
    [ 0.0, 1.0 ],
    [ 1.0, 1.0 ]
];

Targets = [ 0, 1, 1, 0 ];
    
X = DataSet[0];

T = Targets[0];

W1 = [
 [ 0.5,  0.1  ],
 [ 0.2,  -0.4 ],
 [ 0.1,  0.2  ],
 [ -0.5, 0.5  ]
];

W2 = [ 0.2,  0.4, -0.3, 0.5 ];

b1 = [
 -0.4,
 -0.1,
 -0.2,
 0.1
];

b2 = [
 .3
];

def sigmoid(z):
    return 1.0 / (1.0 + exp(-z));

def getZ1(weights, bias, A):
    Z1 = [
        [ weights[0][0] * A[0] + weights[0][1] * A[1] + bias[0]],
        [ weights[1][0] * A[0] + weights[1][1] * A[1] + bias[1]],
        [ weights[2][0] * A[0] + weights[2][1] * A[1] + bias[2]],
        [ weights[3][0] * A[0] + weights[3][1] * A[1] + bias[3]],
    ];
    return Z1;

def getA1(Z1):
    A1 = [
       sigmoid(Z1[0][0]),
       sigmoid(Z1[1][0]),
       sigmoid(Z1[2][0]),
       sigmoid(Z1[3][0])
    ];
    return A1;
    
def getZ2(weights, bias, A):
    Z2 = [
        weights[0] * A[0] + weights[1] * A[1] + weights[2] * A[2] + weights[3] * A[3]+ bias[0]
    ];
    return Z2;

def getA2(Z2):
    return 1.0 / (1.0 + exp(-Z2[0]));

def MSE(A2, T):
    return (A2 - T) ** 2;

def gradientJwrtY(Y, T):
    return 2*(Y - T);

def gradientYwrtZ2(Y):
    return Y * ( 1 - Y );


#Forward Propagation
Z1 = getZ1(W1, b1, X);

print(Z1);

A1 = getA1(Z1);
Z2 = getZ2(W2, b2, A1);
print(Z2);


A2 = getA2(Z2);
J = MSE(A2, T);
#Back Propagation
dJdY = gradientJwrtY(A2, T);
dYdZ2 = gradientYwrtZ2(A2);
dJdZ2 = dJdY * dYdZ2;
dJdW211 = dJdZ2 * A1[0];
dJdW212 = dJdZ2 * A1[1];
dJdW213 = dJdZ2 * A1[2];
dJdW214 = dJdZ2 * A1[3];
dJdB2 = dJdZ2;
dJdZ11 = (W2[0] * dJdZ2 ) * sigmoid(Z1[0][0]) * ( 1.0 - sigmoid(Z1[0][0]) ); 
dJdZ12 = (W2[1] * dJdZ2 ) * sigmoid(Z1[0][1]) * ( 1.0 - sigmoid(Z1[0][1]) ); 
dJdZ13 = (W2[2] * dJdZ2 ) * sigmoid(Z1[0][2]) * ( 1.0 - sigmoid(Z1[0][2]) ); 
dJdZ14 = (W2[3] * dJdZ2 ) * sigmoid(Z1[0][3]) * ( 1.0 - sigmoid(Z1[0][3]) ); 
