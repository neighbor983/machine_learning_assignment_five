from math import exp


DataSet = [
    [ 0, 0 ],
    [ 1,  0 ],
    [ 0, 1 ],
    [ 1,  1 ]     
];

Targets = [0, 1, 1, 0];

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
 [ -.5, .5  -.1 ],
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

for runs in range(1):
    for i in range(4):
        X = DataSet[i];
    
        T = Targets[i];
        
        #forward Pass
        Z1_1 = X[0] * W1[0][0] + X[1] * W1[0][1] + b1[0];
        Z1_2 = X[0] * W1[1][0] + X[1] * W1[1][1] + b1[1];
        Z1_3 = X[0] * W1[2][0] + X[1] * W1[2][1] + b1[2];
        
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
        
        #Back Propagation
        dJdY = GradJwrtY(A3[0][0], T);
        dYdZ3_1 = sigmoidDer(A3_1[0][0]);
        dJdZ3_1 = dJdY * dYdZ3_1;