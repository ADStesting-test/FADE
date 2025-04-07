# Testing the Fault-Tolerance of Multi-Sensor Fusion Perception in Autonomous Driving Systems


# The link for experiment video is:
```
https://zenodo.org/records/14015455
```

The generation approach requires the following dependencies to run:

	1. SORA-SVL simulator: https://github.com/YuqiHuai/SORA-SVL
	2. Apollo autonomous driving platform: https://github.com/ApolloAuto/apollo

# Prerequisites

* A 24-core processor and 32GB memory minimum
* Ubuntu 20.04 or later
* Python 3.8.10 or higher
* NVIDIA graphics card: NVIDIA proprietary driver (>=535.0) must be installed
* Docker-CE version 27.3.1
* NVIDIA Container Toolkit

# Requirements

Install LGSVL PythonAPI (pip3 install): https://github.com/lgsvl/PythonAPI

Version: SVL 2021.3

Other requirements: see in requirements.txt

# Apollo
Website of Apollo: https://apollo.auto/

Installation of Apollo 7.0: https://gitee.com/ApolloAuto/apollo/tree/v7.0.0

map: please put the "SanFrancisco-bin" and "Borregas Ave" in the map folder of Apollo (modules/map/data)

# Run

1. Move the code

   You'll need to put the apollo/myTest folder in your code into the  apollo7/modules, and the SVL/Mytest folder in the code into the .../lgsvl/PythonAPI directory where you have installed.Once you've moved the code, you'll need to recompile apollo.

2. Start the simulator and Apollo

3. Modify the channel

   For the camera channel, you need to modify the output channel of the camera sensor of the corresponding vehicle in the SORA-LGSVL database from /apollo/sensor/camera/front_6mm/image/compressed to /apollo/sensor/zlding/camera/front_6mm/image/compressed；

   For the lidar channel, you need to modify the output channel of the camera sensor of the corresponding vehicle in the SORA-LGSVL database from /apollo/sensor/lidar128/compensator/PointCloud2 to /apollo/sensor/zlding/lidar128/compensator/PointCloud2.

4. Run Camera or Lidar

   For the LiDAR channel, you need to go into Apollo7 and run

   ```
   `bazel run //modules/myTest:camera`
   ```

   For the camera channel, you need to go into apollo7 and run

   ```
   bazel run //modules/myTest:lidar
   ```

5. Run scene

   Once the above is done, the channel will be established and you can start running the scene to test the fault injection effect. Go to.../lgsvl/PythonAPI/Mytest and run the main.py file.

   ```
   python main.py
   ```

   

