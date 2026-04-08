# Julia Set Visualization

This project leverages the computing capabilities of the GPU using the **AMD HIP API** to generate visualizations of an arbitrary **Julia set**.

!(/assets/images/julia_set_picture.png)
Example picture currently implemented in the code

---
Instructions:

1. You can manually adjust the iterations that the algorithm calculates to determine wether a point is convergent.

2. The picture is always centered at (0,0), size_x and size_y determine the x- or y-dimension respectively where the rightmost point is (size_x / 2,R) and the apex of the generated picture is (R, size_y / 2).

3. You may also alter the step variable in order to change the pixel size of the generated picture or the fineness of the picture.

4. The generated picture is stored as a ppm file in the current directory.

---

I tested this program using a RX7900 XT GPU and Ubuntu.
