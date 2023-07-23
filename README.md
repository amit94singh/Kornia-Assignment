##  Task

Take the loftr inference pipeline from kornia and dockerize and make it ready for deployment. https://kornia.readthedocs.io/en/latest/_modules/kornia/feature/loftr/loftr.html


The api created should take two images for matching and return a path to an output image that visualizes matching done on both the images side by side.


Artifacts needed for evaluation:

1. Entire codebase with inference script to GitHub with links to access

2. Saved exported file (use any model repo framework or google drive)

3. Dockerfile to build docker image to launch the inference as a REST endpoint (in the same git)

4. Screenshot of the post request done locally using postman or curl Criteria:

1. Low latency for inference

2. Low inference docker image size

3. Code structure

