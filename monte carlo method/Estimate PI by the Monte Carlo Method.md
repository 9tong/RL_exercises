#RL #MC 

Using the Monte Carlo method to estimate the PI involves randomly sampling points in a unit square. This is a very simple experiment using the classic MC method.

# Method
Randomly generate points within the **unit square** and calculate the ratio of points that fall inside the **unit circle** to the total number of points.
1. Sample $(x, y)$ from the unit square uniformly,
2. check if $x^2 + y^2 \leq 1$, if true, count the point as inside the circle, else count the point as outside the circle,
3. Calculate the ratio $r$ of points inside the circle to the total number of points,
4. $r \approx \frac {\pi}{4}$, thus multiply $r$ by 4 to estimate the value of $\pi$.  
![|300](Figure_pi.png)
