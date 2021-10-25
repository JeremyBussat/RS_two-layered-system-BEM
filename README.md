# RS_two-layered-system-BEM

## Description
The aim of this project is to calculate stress and strain fields in a two layered semi-infinite half space by applying Boundary Element Method. The project includes two python codes. The file "BEM_2D.py" implements the boudary element method and has to be upload as a function. The file "final results.py" implements the method to the problem and solve the tangential stress field.

## Getting started
The code can be run with Python 3.8
The following libraries are required: Numpy, Matplotlib

## Expected output and key parameter
Running "final results.py" will output a window with 9 colormesh figures defining the maximum tangetial stress in given conditions.
Parameters that can be changed in "final results.py" are "Etest" (line10) which defines the Young modulus of the first layer, "Htest" (line11) which defines de height of the first layer,"n" (line13) which defines the number of elements considered in the BEM 

## Authors
Son Pham-Ba - son.phamba@epfl.ch
Jeremy Bussat - jeremy.bussat@epfl.ch

## Version history
0.1 Initial Release

## License
This project is under the EPFL License.
