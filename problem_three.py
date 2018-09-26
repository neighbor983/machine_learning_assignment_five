from math import exp
from plot_helper import cost_run_plot

alpha = .01;

DataSet = [
    [ 0, 0 ],
    [ 1,  0 ],
    [ 0, 1 ],
    [ 1,  1 ]     
];

Targets = [0, 1, 1, 0];

costList = [];
count = 0;


'''
        l11
X1                  l21
        l12                     Y
X2                  l22    
        l13
'''

W1 = [
 [ .5,  .1  ],
 [ .2,  -.4 ],
 [ .1,  .2  ]
];

W2 = [
 [ -.5, .5,  -.1 ],
 [ .2,  .4, -.3 ]
];

W3 = [ .3, -.2];

b1 = [
 [ -.4 ],
 [ -.1 ],
 [ -.2 ]
];

b2 = [ .3, .4 ];

b3 = [ -.2 ];

def sigmoid(z):
    return 1.0  / ( 1.0 + exp(-z) );

def sigmoidDer(y):
    return  y * ( 1.0 - y );
    
def MSE(y, t):
    return ( y - t ) ** 2;
    
def GradJwrtY(y, t):
    return 2 * ( y - t);
    

'''
        l11
X1                  l21
        l12                     Y
X2                  l22    
        l13
'''

for runs in range(20):
    for i in range(4):
        X = DataSet[i];
    
        T = Targets[i];
        
        #forward Pass
        Z1_1 = X[0] * W1[0][0] + X[1] * W1[0][1] + b1[0][0];
        Z1_2 = X[0] * W1[1][0] + X[1] * W1[1][1] + b1[1][0];
        Z1_3 = X[0] * W1[2][0] + X[1] * W1[2][1] + b1[2][0];
        
        Z1 = [
            [ Z1_1 ],
            [ Z1_2 ],
            [ Z1_3 ]
        ];
        
        A1_1 = sigmoid(Z1_1);
        A1_2 = sigmoid(Z1_2);
        A1_3 = sigmoid(Z1_3);
        
        A1 = [
            [ A1_1 ],
            [ A1_2 ],
            [ A1_3 ]
        ];
        
        Z2_1 = A1_1 * W2[0][0] + A1_2 * W2[0][1] + A1_3 * W2[0][2] + b2[0];
        Z2_2 = A1_1 * W2[1][0] + A1_2 * W2[1][1] + A1_3 * W2[1][2] + b2[1];
        
        Z2 = [
            [ Z2_1 ],
            [ Z2_2 ]
        ];

        A2_1 = sigmoid(Z2_1);
        A2_2 = sigmoid(Z2_2);
       
        A2 = [
            [ A2_1 ],
            [ A2_2 ]
        ];
        
        Z3_1 = A2_1 * W3[0] + A2_2 * W3[1] + b3[0];
        
        Z3 = [ Z3_1 ];
        
        A3_1 = sigmoid(Z3_1);
        
        A3 = [ [A3_1] ];
        
        J = MSE(A3_1, T);
        
        count += 1;
    
        costList.append({'Count': count, 'Cost': J});
        
        #Back Propagation
        dJdY = GradJwrtY(A3[0][0], T);
        dYdZ3_1 = sigmoidDer(A3_1);
        dJdZ3_1 = dJdY * dYdZ3_1;
        
        dJdW3_11 = dJdZ3_1 * A2[0][0];
        dJdW3_12 = dJdZ3_1 * A2[1][0];
        
        dJdB3_1 = dJdZ3_1;
        
        dJdZ2_1 = ( W3[0] * dJdZ3_1 + W3[1] * dJdZ3_1) * sigmoidDer(Z2[0][0]);
        dJdZ2_2 = ( W3[0] * dJdZ3_1 + W3[1] * dJdZ3_1) * sigmoidDer(Z2[1][0]);
        
        dJdW2_11 = dJdZ2_1 * A1[0][0];
        dJdW2_12 = dJdZ2_1 * A1[1][0]; 
        dJdW2_13 = dJdZ2_1 * A1[2][0]; 
        dJdW2_21 = dJdZ2_2 * A1[0][0];
        dJdW2_22 = dJdZ2_2 * A1[1][0]; 
        dJdW2_23 = dJdZ2_2 * A1[2][0]; 
        
        dJdB2_1 =  dJdZ2_1;
        dJdB2_2 =  dJdZ2_2;
        
        dJdZ1_1 = ( W2[0][0] * dJdZ2_1 + W2[1][0] * dJdZ2_2 ) * sigmoidDer(Z1_1);
        dJdZ1_2 = ( W2[0][1] * dJdZ2_1 + W2[1][1] * dJdZ2_2 ) * sigmoidDer(Z1_2);
        dJdZ1_3 = ( W2[0][2] * dJdZ2_1 + W2[1][2] * dJdZ2_2 ) * sigmoidDer(Z1_3);
        
        dJdW1_11 = dJdZ1_1 * X[0];
        dJdW1_12 = dJdZ1_1 * X[1];
        dJdW1_21 = dJdZ1_2 * X[0];
        dJdW1_22 = dJdZ1_2 * X[1];
        dJdW1_31 = dJdZ1_3 * X[0];
        dJdW1_32 = dJdZ1_3 * X[1];
        
        dJdB1_1 = dJdZ1_1;
        dJdB1_2 = dJdZ1_2;
        dJdB1_3 = dJdZ1_3;
        
        
        #Update weights and bias
        W1[0][0] = W1[0][0] - alpha * dJdW1_11;
        W1[0][1] = W1[0][1] - alpha * dJdW1_12;
        W1[1][0] = W1[1][0] - alpha * dJdW1_21;
        W1[1][1] = W1[1][1] - alpha * dJdW1_22;
        W1[2][0] = W1[2][0] - alpha * dJdW1_31;
        W1[2][1] = W1[2][1] - alpha * dJdW1_32;

        b1[0][0] = b1[0][0] - alpha * dJdB1_1;
        b1[1][0] = b1[1][0] - alpha * dJdB1_2;
        b1[2][0] = b1[2][0] - alpha * dJdB1_3;
        
        
        W2[0][0] = W2[0][0] - alpha * dJdW2_11;
        W2[0][1] = W2[0][1] - alpha * dJdW2_12;
        W2[0][2] = W2[0][2] - alpha * dJdW2_13;
        W2[1][0] = W2[1][0] - alpha * dJdW2_21;
        W2[1][1] = W2[1][1] - alpha * dJdW2_22;
        W2[1][2] = W2[1][2] - alpha * dJdW2_23;
        
        b2[0] = b2[0] - alpha * dJdB2_1;
        b2[1] = b2[1] - alpha * dJdB2_2;
        
        W3[0] = W3[0] - alpha * dJdW3_11;
        W3[1] = W3[1] - alpha * dJdW3_12;
        
        b3[0] = b3[0] - alpha * dJdB3_1;
        
cost = [];
iteriations = [];

for item in costList:
    cost.append(item['Cost']);
    iteriations.append(item['Count']);

cost_run_plot(cost, iteriations, 'XOR Sigmoid Two Hidden Layers', 'problem_three.svg')


