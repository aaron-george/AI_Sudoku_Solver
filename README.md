
# **Sudoku Solver using OpenCV**

A sudoku solver that can solve sudoku from an image built on python using OpenCV, Tensorflow and Tkinter

Contributors:
* [Nikhil Prabhakar](https://github.com/Nikhil-Prabhakar2806)
* [Ashish Bhatia]( https://github.com/AshishB29)


**To run the program**
* Install [Python3](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjt4aDrl_LoAhX5yDgGHRcuBKMQFjAAegQIChAC&url=https%3A%2F%2Fwww.python.org%2Fdownloads%2F&usg=AOvVaw3VuYRIaaa-SL5nRa6pfny0)

* Install required libraries
* Run solve.py

**Required Libraries**
* [Numpy](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwia1YSto_LoAhUr4jgGHZ9kCCYQFjAAegQIBRAB&url=https%3A%2F%2Fnumpy.org%2F&usg=AOvVaw3L2i9HVc9ZeynETpNrPxO-)
* [OpenCV](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwj9_6bdo_LoAhVPwjgGHWDjDzoQFjAAegQIIhAC&url=https%3A%2F%2Fopencv.org%2F&usg=AOvVaw0nLWFztJIlbNMAYoheT9Qm)
* [Matplotlib](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjyrYvso_LoAhVlyjgGHXLKCA8QFjAAegQIEBAC&url=https%3A%2F%2Fmatplotlib.org%2F&usg=AOvVaw0YgZr7XYZzco6IDaGIE2sw)
* [Keras](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjipOP4o_LoAhW_wzgGHQjFBW8QFjAAegQIARAB&url=http%3A%2F%2Fkeras.io%2F&usg=AOvVaw330NFtOAF1xcgasnbQvfe5)
* [Tensorflow=1.5](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwi_ncmIpPLoAhW8zzgGHTlsB5oQFjAAegQIIRAC&url=https%3A%2F%2Fwww.tensorflow.org%2F&usg=AOvVaw0TGZBeXHx2CVPI2FiDZclR)

**Flow**
* Provide path to the sudoku image<br/>
![Fig A.1.1.Original Image](https://github.com/aaron-george/AI_Sudoku_Solver/blob/master/Screenshots/Screenshot%20from%202020-04-18%2019-35-04.png  )

* Click on Add recognised numbers<br/>
![Fig A.1.1.Original Image](https://github.com/aaron-george/AI_Sudoku_Solver/blob/master/Screenshots/Screenshot%20from%202020-04-18%2019-36-51.png)

* Manually make changes to missing/wrong numbers<br/>
![Fig A.1.1.Original Image](https://github.com/aaron-george/AI_Sudoku_Solver/blob/master/Screenshots/Screenshot%20from%202020-04-18%2019-37-03.png)

* Click on Solve to see the magic<br/>
![Fig A.1.1.Original Image](https://github.com/aaron-george/AI_Sudoku_Solver/blob/master/Screenshots/Screenshot%20from%202020-04-18%2019-37-14.png)


**Steps Involved**

1. Extracting Sudoku from the image
2. Converting the sudoku to numbers
3. Completing the sudoku using BackTracking Algorithm

**1.Extracting sudoku from the image**

**1.1.Preprocessing of the image**

Gaussian blur is done to remove the noise in the image. Kernel size of (9,9).
![Fig A.1.1.Original Image](https://github.com/aaron-george/quarantine-project/blob/master/Capturing-Image/sudoku.jpeg   )               
Fig 1.1.1.Original Image                               
         

       

Adaptive Thresholding is done so that the algorithm decides a threshold based on the pixel values of the neighbouring pixels.The threshold value is a gaussian-weighted sum of the neighbourhood values minus the constant C.
[Read more](https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)


![Fig A.1.1.Original Image](https://github.com/aaron-george/quarantine-project/blob/master/Capturing-Image/Screenshot%20from%202020-04-14%2023-45-02.png  )
                                           <br />  Figure 1.1.3. Adaptive Thresholding<br /> <br /> 
![Fig A.1.1.Original Image](https://github.com/aaron-george/quarantine-project/blob/master/Capturing-Image/Screenshot%20from%202020-04-14%2023-45-10.png)
<br /> Figure 1.1.4. Inversion and Dilation
              
Inversion of colours is performed using Bitwise NOT operation. Dilation of lines is done to increase the thickness of the lines. This concludes the pre-processing of the image.

**1.2  Finding corners of the largest rectangle**

In order to separate the sudoku box apart from the extra junk that is of no use to us, we need to find the corners of the largest rectangle. In order to do so, we use the findContours() function to determine all the contours of the image and find the largest area among them.
[Read more](https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html)

Once the largest area of contours are determined, the combination of the X and Y coordinates from the contour will give the four corners of the rectangle.
Bottom-right point has the largest (x + y) value. Top-left has the smallest (x + y) value. The bottom-left point has the smallest (x — y) value. The top-right point has the largest (x — y) value.

**1.3 Cropping the image**

Crop the image to the square having the sudoku. We describe the square with the distance calculated from the corner points. Then warp the image to that distance after comparing values. Finally, crop the image using transformation.

![Fig A.1.1.Original Image](https://github.com/aaron-george/quarantine-project/blob/master/Capturing-Image/Screenshot%20from%202020-04-14%2023-45-20.png)

Fig 1.3.1 Cropped Image


**1.4. Split the image into 81 cells and infer it** <br /><br /><br />
**1.5. Extract digits from the squares.**<br />
![Fig A.1.1.Original Image](https://github.com/aaron-george/quarantine-project/blob/master/Capturing-Image/gau_sudoku3.jpg)


Fig 1.5.1 Extracted Sudoku Image

**B.Extracting numbers from the squares**

B.1 Splitting the image into 81 squares

B.2 Creating the CNN model using MNIST Database

B.3 Storing the model as .json file

B.4 Running each of the squares to detect the number

B.5 Parse it into a numpy array


**C. Completing the sudoku using BackTracking Algorithm**

The simple backtracking algorithm is used to solve sudoku:

Read more at: https://www.geeksforgeeks.org/sudoku-backtracking-7/

The GUI interface is made using Tkinter library








