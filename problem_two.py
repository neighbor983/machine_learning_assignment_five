from math import exp

DataSet = [
[ -1, -1 ],
[ 1,  -1 ],
[ -1, 1 ],
[ 1,  1 ]     
];

Targets = [-1, 1, 1, -1];

W1 = [
 [ .5,  .1  ],
 [ .2,  -.4 ],
 [ .1,  .2  ],
 [ -.5, .5  ]
];

W2 = [
    .2,  .4, -.3, .5
];

b1 = [
 [ -.4 ],
 [ -.1 ],
 [ -.2 ],
 [ .1  ]
];

b2 = [ .3 ];

def getZ1(weights, bias, x):
    return weights[0] * x[0] + weights[1] * x[1] + bias; 

def getZ2(weights, bias, x):
    return weights[0] * x[0][0] + weights[1] * x[1][0] + weights[2] * x[2][0] + weights[3] * x[3][0] + bias[0];

def bipolarSigmoid(z):
    return ( 1 - exp(-z) ) / ( 1 + exp(-z) );
    
def MSE(A2, T):
    return (A2 - T) ** 2;

def GradJwrtY(y, t):
    return 2 * (y - t);
    
def GradYwrtZ(y):
    return bipolarSigmoidDer(y);
    
def bipolarSigmoidDer(y):
    return  1 / 2.0 * ( 1 - y ) * ( 1 + y );

#Forward Pass

X = DataSet[0];

T = Targets[0];

Z1 = [
 [ getZ1(W1[0], b1[0][0], X) ],
 [ getZ1(W1[1], b1[1][0], X) ],
 [ getZ1(W1[2], b1[2][0], X) ],
 [ getZ1(W1[3], b1[3][0], X) ]
];

A1 = [
 [ bipolarSigmoid(Z1[0][0]) ],
 [ bipolarSigmoid(Z1[1][0]) ],
 [ bipolarSigmoid(Z1[2][0]) ],
 [ bipolarSigmoid(Z1[3][0]) ]
];

Z2 = [
    getZ2(W2, b2, A1)
];

A2 = [
 bipolarSigmoid(Z2[0])
];

J = MSE(A2[0], T);

#BackProp
alpha = .1;
dJdY = GradJwrtY(A2[0], T);
dYdZ2 = GradYwrtZ(A2[0]);
dJdZ2 = dJdY * dYdZ2;

dJdW211 = dJdZ2 * A2[0];
dJdB2 = dJdZ2;

dJdZ11 = (W2[0] * dJdZ2)*bipolarSigmoidDer(Z2[0]);
dJdZ12 = (W2[1] * dJdZ2)*bipolarSigmoidDer(Z2[0]);
dJdZ13 = (W2[2] * dJdZ2)*bipolarSigmoidDer(Z2[0]);
dJdZ14 = (W2[3] * dJdZ2)*bipolarSigmoidDer(Z2[0]);

dJdW111 = dJdZ11 * A1[0][0];
dJdW112 = dJdZ11 * A1[1][0];
dJdW121 = dJdZ12 * A1[0][0];
dJdW122 = dJdZ12 * A1[1][0];
dJdW131 = dJdZ13 * A1[0][0];
dJdW132 = dJdZ13 * A1[1][0];

print(dJdW122);
