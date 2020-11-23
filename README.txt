-------------------------------NOTE---------------------------------
INCASE THE FOLLOWING STEPS DO NOT WORK FOR YOU ON YOUR MACHINE , THEN FEEL FREE TO JUST CHECK OUT "demo.avi" , WHICH IS A DEMONSTRATION OF THIS PROGRAM



Install the required dependencies for this project :
1. Open up cmd in this directory and enter this command		
		pip install -r requirements.txt

This should download and install all dependencies for this project
The C/C++ files are already compiled so no need to mess with them 

In the root directory enter the following command to start the project :

python run_webcam.py --camera 0 --resize 432x368 --model mobilenet_thin

This should start up the camera and show estimations 

If this doesnt work , check your tensorflow version (it only works on tensorflow 1.x)
or
download graph_opt.pb from models/graph/cmu (its about 200mb) , although this shouldn't be a problem unless you're changing parameters from the ArguemntParser



You can also run the other files such as run.py , run_video.py but use the corresponding paramenters to execute them in the command prompt (check source code to know more)


What this project does :
	1. Detects bad neck posture 
	2. Detects bad back posture
	3. Counts the number of times your hand curled (used for counting how many bicep curls you have done)
	4. Generates an "output.avi" file based on what your webcam captured 
	5. Theres also a "demo.avi" file which demonstrates this 

