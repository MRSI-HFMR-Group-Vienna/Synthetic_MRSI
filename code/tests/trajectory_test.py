### import numpy as np
### import matplotlib.pyplot as plt
###
###
### def sft2_Operator(InTraj, OutTraj, Ift_flag):  # 2 is just the name ---> matrix which you can interpred as operator * with image
###
###
### # Determine Exponent of exp based on Ift_flag
### def sft2_Operator(InTraj, OutTraj, Ift_flag, *InputSize):
###     # Assuming InputSize, Ift_flag, InTraj, and OutTraj are defined elsewhere in your script
###     if InputSize:
###         # Input Sizes
###         N1 = InputSize[0]
###         N2 = InputSize[1]
###         N1_floor = np.floor(N1 / 2).astype(int)
###         N2_floor = np.floor(N2 / 2).astype(int)
###
###         if not Ift_flag:
###             InScale = [N1, N2]
###
###
###
### ###########################################################################
import matplotlib.pyplot as plt


# Constants
dMaxGradAmpl = 40
NumberOfLaunTrackPoints = 14
NumberOfLoopPoints = 36
NumberOfBrakeRunPoints = 2 # tiny part in the end to remove => CurGrad_x = real(dGradientValues{1}(1:end-NumberOfBrakeRunPoints)); ==> CurGrad_x = np.real(dGradientValues[0][:-NumberOfBrakeRunPoints])
                                                              #CurGrad_y = imag( ... )
                                                              # Repeat 100 times:  | 36 FID points
                                                              # CurGrad_x = cat(2,CurGrad_x,repmat(CurGrad_x(1,NumberOfLaunTrackPoints+1:end),[1 99])); # figure; plot(CurGrad_x)

# Initialize the dGradientValues dictionary to hold complex gradient values
dGradientValues = {}

# Initialize the list with the same number of elements as in the MATLAB script
dGradientValues[1] = [0] * 52

# Assign complex values to each element in the list
dGradientValues[1][0] = complex(0, 0)
dGradientValues[1][1] = complex(0, 0)
dGradientValues[1][2] = complex(0.00799057, -0.00239154)
dGradientValues[1][3] = complex(0.0159811, -0.00478307)
dGradientValues[1][4] = complex(0.0239717, -0.00717461)
dGradientValues[1][5] = complex(0.0319623, -0.00956615)
dGradientValues[1][6] = complex(0.0239717, -0.00717461)
dGradientValues[1][7] = complex(0.0159811, -0.00478307)
dGradientValues[1][8] = complex(0.00799057, -0.00239154)
dGradientValues[1][9] = complex(-2.65902e-18, 7.95831e-19)
dGradientValues[1][10] = complex(0, 0)
dGradientValues[1][11] = complex(0.00234671, 0.00897923)
dGradientValues[1][12] = complex(0.00267846, 0.0171732)
dGradientValues[1][13] = complex(-6.46479e-19, 0.0232919)
dGradientValues[1][14] = complex(-0.00404459, 0.022938)
dGradientValues[1][15] = complex(-0.00796629, 0.0218872)
dGradientValues[1][16] = complex(-0.0116459, 0.0201713)
dGradientValues[1][17] = complex(-0.0149717, 0.0178426)
dGradientValues[1][18] = complex(-0.0178426, 0.0149717)
dGradientValues[1][19] = complex(-0.0201713, 0.0116459)
dGradientValues[1][20] = complex(-0.0218872, 0.00796629)
dGradientValues[1][21] = complex(-0.022938, 0.00404459)
dGradientValues[1][22] = complex(-0.0232919, 1.42622e-18)
dGradientValues[1][23] = complex(-0.022938, -0.00404459)
dGradientValues[1][24] = complex(-0.0218872, -0.00796629)
dGradientValues[1][25] = complex(-0.0201713, -0.0116459)
dGradientValues[1][26] = complex(-0.0178426, -0.0149717)
dGradientValues[1][27] = complex(-0.0149717, -0.0178426)
dGradientValues[1][28] = complex(-0.0116459, -0.0201713)
dGradientValues[1][29] = complex(-0.00796629, -0.0218872)
dGradientValues[1][30] = complex(-0.00404459, -0.022938)
dGradientValues[1][31] = complex(-2.85243e-18, -0.0232919)
dGradientValues[1][32] = complex(0.00404459, -0.022938)
dGradientValues[1][33] = complex(0.00796629, -0.0218872)
dGradientValues[1][34] = complex(0.0116459, -0.0201713)
dGradientValues[1][35] = complex(0.0149717, -0.0178426)
dGradientValues[1][36] = complex(0.0178426, -0.0149717)
dGradientValues[1][37] = complex(0.0201713, -0.0116459)
dGradientValues[1][38] = complex(0.0218872, -0.00796629)
dGradientValues[1][39] = complex(0.022938, -0.00404459)
dGradientValues[1][40] = complex(0.0232919, -4.27865e-18)
dGradientValues[1][41] = complex(0.022938, 0.00404459)
dGradientValues[1][42] = complex(0.0218872, 0.00796629)
dGradientValues[1][43] = complex(0.0201713, 0.0116459)
dGradientValues[1][44] = complex(0.0178426, 0.0149717)
dGradientValues[1][45] = complex(0.0149717, 0.0178426)
dGradientValues[1][46] = complex(0.0116459, 0.0201713)
dGradientValues[1][47] = complex(0.00796629, 0.0218872)
dGradientValues[1][48] = complex(0.00404459, 0.022938)
dGradientValues[1][49] = complex(5.70486e-18, 0.0232919)
dGradientValues[1][50] = complex(5.70486e-18, 0.0232919)
dGradientValues[1][51] = complex(0, 0.00745853)

# Print the results for verification
for i, val in enumerate(dGradientValues[1]):
    print(f"dGradientValues[1][{i + 1}] = {val}")

#import numpy as np
#plt.plot(np.cumsum())


real_parts = [val.real for val in dGradientValues[1]]
plt.plot(real_parts)
plt.show()