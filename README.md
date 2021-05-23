## What's the challenge?

You are expected to find the 6DoF pose* of the camera in images img2 and img3 w.r.t. pose in reference image img1.


### My Solution

Using SIFT feature tracking, FLANN matching and subsequent calculations of the fundamental and essential matrix. After decomposing the essential matrix finding the right R and T out of four possible combinations, by checking if points are in front of both cameras.
