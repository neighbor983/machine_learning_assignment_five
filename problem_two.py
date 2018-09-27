from math import exp
from plot_helper import cost_run_plot

count = 0;

costList = [];

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

alpha = .1;

def getZ1(weights, bias, x):
    return weights[0] * x[0] + weights[1] * x[1] + bias; 

def getZ2(weights, bias, x):
    return weights[0] * x[0][0] + weights[1] * x[1][0] + weights[2] * x[2][0] + weights[3] * x[3][0] + bias[0];

def bipolarSigmoid(z):
    return ( 1.0 - exp(-z) ) / ( 1.0 + exp(-z) );
    
def MSE(y, T):
    return (y - T) ** 2;

def GradJwrtY(y, t):
    return 2 * (y - t);
    
def GradYwrtZ(y):
    return bipolarSigmoidDer(y);
    
def bipolarSigmoidDer(y):
    return  1.0 / 2.0 * ( 1.0 - y ) * ( 1.0 + y );
    
def activationDer(x):
    return .5 * ( 1.0 + bipolarSigmoid(x) ) * ( 1.0 - bipolarSigmoid(x) );

for runs in range(500):
    for i in range(4):
        A0 = DataSet[i];
    
        T = Targets[i];
    
        #Forward Pass
        Z1_1 = getZ1(W1[0], b1[0][0], A0);
        Z1_2 = getZ1(W1[1], b1[1][0], A0);
        Z1_3 = getZ1(W1[2], b1[2][0], A0);
        Z1_4 = getZ1(W1[3], b1[3][0], A0);

        Z1 = [
        [ Z1_1 ],
        [ Z1_2 ],
        [ Z1_3 ],
        [ Z1_4 ]
    ];
    
        A1_1 = bipolarSigmoid(Z1[0][0]);
        A1_2 = bipolarSigmoid(Z1[1][0]);
        A1_3 = bipolarSigmoid(Z1[2][0]);
        A1_4 = bipolarSigmoid(Z1[3][0]);
    
        A1 = [
            [ A1_1 ],
            [ A1_2 ],
            [ A1_3 ],
            [ A1_4 ]
        ];
    
        Z2_1 =  getZ2(W2, b2, A1);
    
        Z2 = [ Z2_1 ];
    
        A2_1 = bipolarSigmoid(Z2[0]);
    
        A2 = [ A2_1 ];
    
        J = MSE(A2[0], T);
        
        count += 1;

        costList.append({'Count': count, 'Cost': J});
    
        #BackProp
        dJdY = GradJwrtY(A2[0], T);
        dYdZ2_1 = GradYwrtZ(A2[0]);
        dJdZ2_1 = dJdY * dYdZ2_1;
    
        dJdW2_11 = dJdZ2_1 * A1[0][0];
        dJdW2_21 = dJdZ2_1 * A1[1][0];
        dJdW2_31 = dJdZ2_1 * A1[2][0];
        dJdW2_41 = dJdZ2_1 * A1[3][0];
    
        dJdB2_1 = dJdZ2_1;
    
        #Level one
        dJdZ1_1 = (W2[0] * dJdZ2_1) * activationDer(Z1_1);
        dJdZ1_2 = (W2[1] * dJdZ2_1) * activationDer(Z1_2);
        dJdZ1_3 = (W2[2] * dJdZ2_1) * activationDer(Z1_3);
        dJdZ1_4 = (W2[3] * dJdZ2_1) * activationDer(Z1_4);

        dJdW1_11 = dJdZ1_1 * A0[0];
        dJdW1_12 = dJdZ1_1 * A0[1];
        dJdW1_21 = dJdZ1_2 * A0[0];
        dJdW1_22 = dJdZ1_2 * A0[1];
        dJdW1_31 = dJdZ1_3 * A0[0];
        dJdW1_32 = dJdZ1_3 * A0[1];
        dJdW1_41 = dJdZ1_4 * A0[0];
        dJdW1_42 = dJdZ1_4 * A0[1];
        
        dJdB1_1 = dJdZ1_1;
        dJdB1_2 = dJdZ1_2;
        dJdB1_3 = dJdZ1_3;
        dJdB1_4 = dJdZ1_4;
    
        #Update weights and bias
        W2[0] = W2[0] - alpha * dJdW2_11;
        W2[1] = W2[1] - alpha * dJdW2_21;
        W2[2] = W2[2] - alpha * dJdW2_31;
        W2[3] = W2[3] - alpha * dJdW2_41;
    
        b2[0] = b2[0] - alpha * dJdB2_1;

        W1[0][0] = W1[0][0] - alpha * dJdW1_11;
        W1[0][1] = W1[0][1] - alpha * dJdW1_12;
        W1[1][0] = W1[1][0] - alpha * dJdW1_21;
        W1[1][1] = W1[1][1] - alpha * dJdW1_22;
        W1[2][0] = W1[2][0] - alpha * dJdW1_31;
        W1[2][1] = W1[2][1] - alpha * dJdW1_32;
        W1[3][0] = W1[3][0] - alpha * dJdW1_41;
        W1[3][1] = W1[3][1] - alpha * dJdW1_42;

        b1[0][0] = b1[0][0] - alpha * dJdB1_1;
        b1[1][0] = b1[1][0] - alpha * dJdB1_2;
        b1[2][0] = b1[2][0] - alpha * dJdB1_3;
        b1[3][0] = b1[3][0] - alpha * dJdB1_4;

cost = [];
iteriations = [];

for item in costList:
    cost.append(item['Cost']);
    iteriations.append(item['Count']);

cost_run_plot(cost, iteriations, 'XOR Bipolar', 'problem_two.svg')

for i in range(4):
    X = DataSet[i];
    
    T = Targets[i];
    
    #Forward Pass
    Z1_1 = getZ1(W1[0], b1[0][0], X);
    Z1_2 = getZ1(W1[1], b1[1][0], X);
    Z1_3 = getZ1(W1[2], b1[2][0], X);
    Z1_4 = getZ1(W1[3], b1[3][0], X);

    Z1 = [
        [ Z1_1 ],
        [ Z1_2 ],
        [ Z1_3 ],
        [ Z1_4 ]
    ];
    
    A1_1 = bipolarSigmoid(Z1[0][0]);
    A1_2 = bipolarSigmoid(Z1[1][0]);
    A1_3 = bipolarSigmoid(Z1[2][0]);
    A1_4 = bipolarSigmoid(Z1[3][0]);
    
    A1 = [
            [ A1_1 ],
            [ A1_2 ],
            [ A1_3 ],
            [ A1_4 ]
    ];
    
    Z2_1 =  getZ2(W2, b2, A1);
    
    Z2 = [ Z2_1 ];
    
    A2_1 = bipolarSigmoid(Z2[0]);
    
    A2 = [ A2_1 ];
    
    print(str(X)+ ': ' + str(A2_1));

